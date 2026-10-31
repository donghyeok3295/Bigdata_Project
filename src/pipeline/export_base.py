import logging
import time

from src.models.features import load_yaml

log = logging.getLogger(__name__)

TS_FORMAT = "yyyy-MM-dd HH:mm:ss"


def export_base(spark: object, cfg: dict) -> None:
    """코호트에서 최소 열만 골라 시험 연도 전후로 따로 내보낸다. 시각은 서울 시간대 문자열."""
    from pyspark.sql import functions as F

    exp = cfg["export"]
    src = exp["source_cols"]
    t0 = time.time()
    spark.conf.set("spark.sql.session.timeZone", exp["session_timezone"])
    df = spark.read.parquet(exp["cohort_path"])
    n_in = df.count()
    out = df.select(
        F.col(src["call_id"]).alias("call_id"),
        F.date_format(F.col(src["dclr_ts"]), TS_FORMAT).alias("dclr_str"),
        F.col(src["center_id"]).alias("center_id"),
        F.col(src["cntr_nm"]).alias("cntr_nm"),
        F.col(src["sgg"]).alias("sgg"),
        F.col(src["dist_raw"]).cast("double").alias("dist_raw"),
        F.col(src["resp_min"]).cast("double").alias("resp_min"),
        F.col(src["late_srv"]).cast("int").alias("late_srv"),
        F.col(src["unit_type_srv"]).alias("unit_type_srv"),
        F.col(src["distance_invalid"]).cast("int").alias("distance_invalid"),
    ).withColumn("year", F.substring("dclr_str", 1, 4).cast("int"))
    train = out.where(F.col("year") < exp["test_year"])
    test = out.where(F.col("year") >= exp["test_year"])
    train.write.mode("overwrite").partitionBy("year").parquet(exp["out_train"])
    test.write.mode("overwrite").partitionBy("year").parquet(exp["out_test"])
    log.info(
        "export_base in=%d train=%d test=%d sec=%.1f",
        n_in,
        spark.read.parquet(exp["out_train"]).count(),
        spark.read.parquet(exp["out_test"]).count(),
        time.time() - t0,
    )


def main() -> None:
    """CLI: SparkSession을 열어 내보내기를 실행한다."""
    from pyspark.sql import SparkSession

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_yaml("configs/data.yaml")
    spark = (
        SparkSession.builder.appName("export_base")
        .config("spark.sql.session.timeZone", cfg["export"]["session_timezone"])
        .getOrCreate()
    )
    export_base(spark, cfg)


if __name__ == "__main__":
    main()
