from dagster import SourceAsset, TableSchema, asset
from pandas import DataFrame

raw_country_populations = SourceAsset(
    "raw_country_populations",
    metadata={
        "column_schema": TableSchema.from_name_type_dict(
            {
                "country": "string",
                "continent": "string",
                "region": "string",
                "pop2018": "int",
                "pop2019": "int",
                "change": "string",
            }
        ),
    },
)


@asset
def country_populations(raw_country_populations) -> DataFrame:
    country_populations = raw_country_populations.copy()
    country_populations["change"] = (
        country_populations["change"]
        .str.rstrip("%")
        .str.replace("−", "-")  # noqa: RUF001
        .astype("float")
        / 100.0
    )
    return country_populations


@asset
def continent_stats(country_populations: DataFrame) -> DataFrame:
    result = country_populations.groupby("continent").agg({"pop2019": "sum", "change": "mean"})
    return result


@asset
def country_stats(country_populations: DataFrame, continent_stats: DataFrame) -> DataFrame:
    result = country_populations.join(continent_stats, on="continent", lsuffix="_continent")
    result["continent_pop_fraction"] = result["pop2019"] / result["pop2019_continent"]
    return result
