import polars as pl
from utils import set_table_dtypes, convert_strings

def load_and_prepare_data(data_path):
    base_train = pl.read_csv(data_path + "csv_files/train/train_base.csv")
    person_1_train = pl.read_csv(data_path + 'csv_files/train/train_person_1.csv').pipe(set_table_dtypes) 

    applprev_train = pl.concat(
        [
            pl.read_csv(data_path + 'csv_files/train/train_applprev_1_0.csv').pipe(set_table_dtypes),
            pl.read_csv(data_path + 'csv_files/train/train_applprev_1_1.csv').pipe(set_table_dtypes),
        ],
        how="vertical_relaxed",
    )

    static_train = pl.concat(
        [
            pl.read_csv(data_path + "csv_files/train/train_static_0_0.csv").pipe(set_table_dtypes),
            pl.read_csv(data_path + "csv_files/train/train_static_0_1.csv").pipe(set_table_dtypes),
        ],
        how="vertical_relaxed",
    )

    static_cb_train = pl.read_csv(data_path + "csv_files/train/train_static_cb_0.csv").pipe(set_table_dtypes)
    credit_bureau_b_2_train = pl.read_csv(data_path + 'csv_files/train/train_credit_bureau_b_2.csv').pipe(set_table_dtypes)

    base_train = base_train.drop(["date_decision", "WEEK_NUM", "MONTH"])

    with pl.StringCache():
        person_1_train_agg = person_1_train.group_by("case_id").agg(
            pl.col("education_927M").filter(pl.col("num_group1") == 0).cast(pl.String).first().alias("education"),
            pl.col("mainoccupationinc_384A").filter(pl.col("num_group1") == 0).first().alias("main_income"),
            pl.col("sex_738L").filter(pl.col("num_group1") == 0).cast(pl.String).first().alias("sex"),
            pl.col("safeguarantyflag_411L").filter(pl.col("num_group1") == 0).cast(pl.Boolean).first().alias("flag"),
            (2024 - pl.col("birth_259D").dt.year()).first().alias("age")
        )

        applprev_train_agg = applprev_train.group_by("case_id").agg(
            pl.col("cancelreason_3545846M").mode().first().alias("cancelreason_most_frecuent"),
            pl.col("childnum_21L").cast(pl.Int64).max().alias("childnum"),
            pl.col("annuity_853A").cast(pl.Float64).mean().alias("mean_annuity"),
            pl.col("currdebt_94A").mode().first().alias("debt"), 
            pl.col("credacc_credlmt_575A").cast(pl.Int64).max().alias("credit_card_limit"),
            pl.col("credamount_590A").cast(pl.Int64).max().alias("credit_ammount"),
            pl.col("downpmt_134A").cast(pl.Int64).max().alias("downpayment"),
            pl.col("pmtnum_8L").cast(pl.Int64).max().alias("number_of_payments"),    
        ).drop("num_group1")

        selected_static_cb_cols = []
        columns_to_convert_static_cb = [
            "days120_123L",
            "days180_256L",
            "days30_165L",
            "days360_512L",
            "days90_310L",
            "firstquarter_103L",
            "fourthquarter_440L",
            "secondquarter_766L",
            "thirdquarter_1082L",
            "numberofqueries_373L"
        ]

        for col in columns_to_convert_static_cb:
            null_percentage = (static_cb_train[col].null_count() / len(static_cb_train))
            static_cb_train.select(pl.col(col).cast(pl.Int32))
            selected_static_cb_cols.append(col)

        selected_static_cols = []
        columns_to_convert_static = [
            "cntpmts24_3658933L",
            "lastst_736L",
            "mobilephncnt_593L",
            "monthsannuity_845L",
            "numinstlallpaidearly3d_817L",
            "numinstlsallpaid_934L",
            "numinstlswithdpd10_728L",
            "numinstlswithdpd5_4187116L",
            "numinstlswithoutdpd_562L",
            "numinstmatpaidtearly2d_4499204L", 
            "numinstpaid_4499208L",
            "numinstpaidearly3d_3546850L",
            "numinstpaidearly3dest_4493216L", 
            "numinstpaidearly5d_1087L",
            "numinstpaidearly5dest_4493211L",
            "numinstpaidearly5dobd_4499205L",
            "numinstpaidearly_338L",
            "numinstpaidearlyest_4493214L",
            "numinstpaidlastcontr_4325080L",
            "numinstpaidlate1d_3546852L",
            "numinstregularpaid_973L",
            "numinstregularpaidest_4493210L",
            "numrejects9m_859L",
            "pctinstlsallpaidearl3d_427L",
            "pctinstlsallpaidlat10d_839L",
            "pctinstlsallpaidlate1d_3546856L",
            "pctinstlsallpaidlate4d_3546849L",
            "pctinstlsallpaidlate6d_3546844L",
            "pmtnum_254L",
            "twobodfilling_608L"
        ]

        for col in static_train.columns:
            null_percentage = (static_train[col].null_count() / len(static_train))
            if ((col[-1] in ("A", "M","P")) and (null_percentage < 0.3)):
                selected_static_cols.append(col)
            elif ((col in columns_to_convert_static) and (null_percentage < 0.3)):
                if static_train[col].dtype == pl.Utf8:
                    static_train = static_train.with_columns(pl.col(col).cast(pl.Categorical))
                    selected_static_cols.append(col)
                else:
                    static_train.select(pl.col(col).cast(pl.Int32))
                    selected_static_cols.append(col)

        credit_bureau_b_2_train_agg = credit_bureau_b_2_train.group_by("case_id").agg(
            pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_635A_max"),
            (pl.col("pmts_dpdvalue_108P") > 31).max().alias("pmts_dpdvalue_108P_over31")
        )

        for col in credit_bureau_b_2_train_agg.columns:
            null_percentage = (credit_bureau_b_2_train_agg[col].null_count() / len(credit_bureau_b_2_train_agg))
            if ((null_percentage >= 0.3) or (col[-1] == "D")):
                credit_bureau_b_2_train_agg.drop(col)

    data = base_train.join(
        static_train.select(["case_id"] + selected_static_cols), how="left", on="case_id", coalesce=True
    ).join(
        static_cb_train.select(["case_id"] + selected_static_cb_cols), how="left", on="case_id", coalesce=True
    ).join(
        person_1_train_agg, how="left", on="case_id", coalesce=True
    ).join(
        applprev_train_agg, how="left", on="case_id", coalesce=True
    ).join(credit_bureau_b_2_train_agg, how="left", on="case_id", coalesce=True)

    data = convert_strings(data)

    y = data['target']
    X = data.drop(["case_id", "target"])

    NUMERIC_POLARS_DTYPES = [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, 
    ]

    numerical_columns = X.select(pl.col(NUMERIC_POLARS_DTYPES)).columns  
    non_numerical_columns = X.select(pl.exclude(NUMERIC_POLARS_DTYPES)).columns 

    X = X.with_columns([
        pl.col(numerical_columns).fill_null(strategy="mean"),
        *(pl.col(col).fill_null(pl.col(col).mode()) for col in non_numerical_columns)
    ])
    
    X = X.to_pandas()
    y = y.to_pandas()

    return X, y
