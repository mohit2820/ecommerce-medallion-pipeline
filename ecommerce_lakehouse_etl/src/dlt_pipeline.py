import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

# =========================================================================
# 1. BRONZE LAYER: Raw Data Ingestion (Auto Loader)
# =========================================================================
@dlt.table(
    name="bronze.raw_orders_2",  # <--- Explicitly routes to bronze schema
    comment="Raw streaming orders ingested directly from cloud volume CSVs"
)
def raw_orders():
    # Fetch target catalog dynamically or fallback to dev default
    catalog = spark.conf.get("catalog", "ecommerce_catalog_dev")
    volume_path = f"/Volumes/{catalog}/bronze/raw_data_volume/ecommerce_transactions/"
    
    # Auto Loader automatically streams incoming CSV files
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(volume_path)
        .withColumn("ingestion_time", F.current_timestamp())
        .withColumn("source_file", F.col("_metadata.file_path"))
    )

# =========================================================================
# 2. SILVER LAYER: Data Cleaning, Typing & Quality Checks
# =========================================================================
@dlt.table(
    name="silver.silver_orders_2",  # <--- Explicitly routes to silver schema
    comment="Cleaned, validated, and typed e-commerce orders"
)
# Data Quality Expectations: Bad rows are dropped automatically
@dlt.expect_or_drop("valid_order_id", "order_id IS NOT NULL")
@dlt.expect_or_drop("valid_customer_id", "customer_id IS NOT NULL AND customer_id != ''")
@dlt.expect_or_drop("valid_order_date", "order_date IS NOT NULL")
@dlt.expect_or_drop("valid_product_name", "product_name IS NOT NULL")
@dlt.expect_or_drop("valid_quantity", "quantity IS NOT NULL AND quantity > 0")
@dlt.expect_or_drop("valid_unit_price", "unit_price IS NOT NULL AND unit_price > 0")
@dlt.expect_or_drop("valid_status", "status IN ('COMPLETED', 'PENDING', 'CANCELLED')")
def silver_orders():
    return (
        # Read from the specific schema-prefixed Bronze DLT table
        dlt.read_stream("bronze.raw_orders_2")
        
        # String trimming & standardization
        .withColumn("customer_id", F.trim(F.col("customer_id")))
        .withColumn("product_name", F.lower(F.trim(F.col("product_name"))))
        .withColumn("status", F.upper(F.trim(F.col("status"))))
        
        # Explicit data type casting
        .withColumn("order_id", F.col("order_id").cast(IntegerType()))
        .withColumn("quantity", F.col("quantity").cast(IntegerType()))
        .withColumn("unit_price", F.col("unit_price").cast(DoubleType()))
        
        # Parse multiple potential date formats safely
        .withColumn(
            "order_date",
            F.coalesce(
                F.try_to_date(F.col("order_date"), "yyyy-MM-dd"),
                F.try_to_date(F.col("order_date"), "MM/dd/yyyy"),
                F.try_to_date(F.col("order_date"), "dd-MM-yyyy")
            )
        )
        
        # Calculated column
        .withColumn("total_amount", F.round(F.col("quantity") * F.col("unit_price"), 2))
        
        # Deduplicate records by order_id
        .dropDuplicates(["order_id"])
    )

# =========================================================================
# 3. GOLD LAYER: Business KPIs & Aggregations
# =========================================================================

# KPI 1: Daily Sales Performance
@dlt.table(
    name="gold.daily_revenue_kpi_3",  # <--- Explicitly routes to gold schema
    comment="Gold KPI: Daily aggregated revenue, order count, and unique customers"
)
def daily_revenue_kpi():
    return (
        dlt.read("silver.silver_orders_2")
        .filter(F.col("status") == "COMPLETED")
        .groupBy("order_date")
        .agg(
            F.round(F.sum("total_amount"), 2).alias("total_daily_revenue"),
            F.countDistinct("order_id").alias("total_completed_orders"),
            F.countDistinct("customer_id").alias("distinct_active_customers")
        )
    )

# KPI 2: Product Metrics
@dlt.table(
    name="gold.product_performance_kpi_3",  # <--- Explicitly routes to gold schema
    comment="Gold KPI: Product revenue and volume metrics"
)
def product_performance_kpi():
    return (
        dlt.read("silver.silver_orders_2")
        .filter(F.col("status") == "COMPLETED")
        .groupBy("product_name")
        .agg(
            F.sum("quantity").alias("total_units_sold"),
            F.round(F.sum("total_amount"), 2).alias("gross_revenue"),
            F.round(F.avg("unit_price"), 2).alias("avg_selling_price")
        )
    )

# KPI 3: Customer Value (CLV)
@dlt.table(
    name="gold.customer_clv_kpi_3",  # <--- Explicitly routes to gold schema
    comment="Gold KPI: Customer lifetime spend and average order value"
)
def customer_clv_kpi():
    return (
        dlt.read("silver.silver_orders_2")
        .filter(F.col("status") == "COMPLETED")
        .groupBy("customer_id")
        .agg(
            F.countDistinct("order_id").alias("total_completed_orders"),
            F.round(F.sum("total_amount"), 2).alias("customer_lifetime_spend"),
            F.round(F.avg("total_amount"), 2).alias("avg_order_value")
        )
    )