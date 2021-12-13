# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:54:26 2021

@author: whetz
"""

import pandas as pd
import numpy as np
import pymc3 as pm
import graphviz
import arviz as az
import seaborn as sns
from scipy.stats import norm


def read_data(predictors, path="fred-data-pres\\pres-data.csv"):
    df = pd.read_csv(path, index_col=0).reset_index().query("year < 2020")
    df = df.rename(
        {
            "Housing Inventory: Median Listing Price": "med_housing",
            "Unemployment Rate": "unemployment",
            "All Employees: Federal Government	Housing Inventory: Median Home Size in ": "govt_employees",
            "Housing Inventory: Median Days on Market": "housing_mkt_days",
            "Housing Inventory: New Listing Count": "housing_listings",
            "All Employees: Financial Activities": "financial_act",
            "All Employees: Education and Health Services": "educ_health",
            "All Employees: Federal Government": "fed_employees",
            "Resident Population": "population",
            "Per Capita Personal Income": "income",
            "Housing Inventory: Median Home Size in Square Feet": "home_size",
        },
        axis=1,
    )

    df.insert(0, "year_month", df.year + df.month / 100)

    id_cols = ["year_month", "year", "month", "city", "state"]
    id_cols.extend(predictors)
    df = df[id_cols]

    return df


def generate_time_data(df, city, order, predictors):

    df = df.query(f"city== '{city}'").reset_index(drop=True)

    df_input = pd.DataFrame()

    year_months = []

    for period in range(order, len(df)):
        year_month = df.year_month.loc[period]
        # print(year_month)
        df2 = df.loc[period - order : period - 1]
        # print(df2)

        df2["step"] = [i for i in range(order)]
        df_pivot = df2.pivot(
            index="city", columns="step", values=predictors
        ).reset_index()
        df_pivot.columns = [col[0] + "_" + str(col[1]) for col in df_pivot.columns]
        # print(df_pivot)

        # Get relevant outputs
        query = f"year_month == {year_month}"
        for col in predictors:
            df_pivot[col] = df.query(query).reset_index(drop=True)[col].iloc[0]
        df_input = pd.concat([df_input, df_pivot])
        year_months.append(year_month)
    # print(df_input)

    df_input.insert(1, "year_month", year_months)
    df_input = df_input.reset_index(drop=True)
    # df_input.columns=[str(col[0]) + str(col[1]) for col in df_input.columns.values]
    return df_input


def write_model_module(input_df, sig=1000000):

    nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    columns = [col for col in input_df.columns if col[-1] in nums]
    outputs = [col for col in input_df.columns[2:] if col[-1] not in nums]

    with open("model_builder.py", "w") as writer:
        writer.write("import pymc3 as pm\n")
        writer.write("def build_model(input_df, sig=1000000):\n")
        writer.write("\twith pm.Model() as model_comb:\n")

        for output in outputs:
            writer.write(f"\t\t# {output}\n")
            input_names = []
            for col in columns:
                input_names.append(f"\t{col}_{output}\n")
                writer.write(
                    f"\t\t{col}_{output} = pm.Normal(name='{col}_{output}', mu=0, sigma=sig)\n"
                )
                # print("")
            writer.write("\n")
            writer.write(f"\t\ttheta_{output} = (\n")
            for name in columns:
                if name == columns[-1]:
                    writer.write(f"\t\t\t{name}_{output}*input_df.{name}\n")
                else:
                    writer.write(f"\t\t\t{name}_{output}*input_df.{name}+\n")
            writer.write("\t\t)\n")

            writer.write(
                f"\t\t{output} = pm.Normal('{output}', theta_{output}, sd={sig}, observed=input_df.{output})\n"
            )

            writer.write("\t\n")
        writer.write("\treturn model_comb")


def generate_advi_posterior(model_comb):
    SEED = 12345
    np.random.seed(SEED)

    # Obtain the variational approximation to the beta
    with model_comb:
        approx = pm.fit(50000, method="advi", random_seed=SEED)
    advi = approx.sample(50000)
    values = advi.varnames
    means = []
    stds = []

    for value in advi.varnames:
        means.append(round(np.mean(advi.get_values(value)), 3))
        stds.append(round(np.std(advi.get_values(value)), 5))
    parameters = pd.DataFrame({"variable": values, "mean": means, "std": stds})

    return parameters


def get_initial_prediction(input, year_month, order):

    nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    outputs = [col for col in input.columns[2:] if col[-1] not in nums]
    new_df = input.query(f"year_month == {year_month}").reset_index(drop=True)

    city = new_df["city_"].loc[0]
    build_df = pd.Series()

    for col in outputs:
        for i in range(order, 0, -1):

            # print(i)

            if i == order:
                new_col = f"{col}_{i-1}"
                build_df[new_col] = new_df[col].loc[0]
            else:
                new_col = f"{col}_{i-1}"
                build_df[new_col] = new_df[f"{col}_{i}"].loc[0]
    build_df = pd.DataFrame(build_df).transpose()
    build_df.insert(0, "city_", city)
    build_df.insert(1, "year_month", year_month)
    return build_df


def get_new_prediction(
    df_preds, input_df, parameters, samples, order, df_stdev=pd.DataFrame(), first=True
):
    nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    outputs = [col for col in input_df.columns[2:] if col[-1] not in nums]
    # print(df_preds)
    city = df_preds["city_"].iloc[-1]
    year_month = df_preds["year_month"].iloc[-1]

    month = int(round(100 * (year_month % 1), 0))
    # print(year_month)
    year = round(year_month, 0)
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1
    # print(month)

    year_month = year + month / 100

    df_new = pd.Series()
    df_std = pd.Series()

    for output in outputs:
        # print(output)
        param_list = list(
            parameters[parameters.variable.str.contains(f"_{output}")].variable
        )
        sample_array = np.zeros(samples)

        for param in param_list:

            num = int(param.split(f"_{output}")[0][-1])

            if num < order and num > 0:
                # print("Num", num)
                # print(F"{output}_{str(num)}")
                df_new[f"{output}_{str(num-1)}"] = df_preds[
                    f"{output}_{str(num)}"
                ].iloc[-1]
                # print(df_pred[F"{output}_{str(num)}"].iloc[-1])
            mean = (
                parameters.query(f"variable == '{param}'")["mean"]
                .reset_index(drop=True)
                .loc[0]
            )
            std = (
                parameters.query(f"variable == '{param}'")["std"]
                .reset_index(drop=True)
                .loc[0]
            )

            if first or param.split(f"_{output}")[0] not in df_stdev.columns:
                # print(param.split(F"_{output}"))
                sample_array += (
                    np.random.normal(loc=mean, scale=std, size=samples)
                    * df_preds[param.split(f"_{output}")[0]].iloc[-1]
                )
            else:
                sample_array += np.random.normal(
                    loc=mean, scale=std, size=samples
                ) * np.random.normal(
                    loc=df_preds[param.split(f"_{output}")[0]].iloc[-1],
                    scale=df_stdev[param.split(f"_{output}")[0]].iloc[-1],
                )
        new_average = round(np.mean(sample_array), 2)
        new_std = round(np.std(sample_array), 2)

        df_new[f"{output}_{order-1}"] = new_average
        df_std[f"{output}_{order-1}"] = new_std
    df_new = pd.DataFrame(df_new).transpose()
    df_new.insert(0, "city_", city)
    df_new.insert(1, "year_month", year_month)
    df_new

    df_std = pd.DataFrame(df_std).transpose()
    df_std.insert(0, "city_", city)
    df_std.insert(1, "year_month", year_month)

    # print(list(df_stdev.columns[2:]))
    for col in list(df_stdev.columns[2:]):
        new_num = int(col[-1]) - 1
        if new_num >= 0:
            new_col = col.replace(col[-1], str(new_num))
            # print(df_stdev[col].iloc[-1])
            df_std[new_col] = df_stdev[col].iloc[-1]
    df_std
    return df_new, df_std


def run_projections(order, input, start_year_month, steps, parameters, samples=1000000):

    order = 4
    df_pred = get_initial_prediction(input=input, year_month=2017.12, order=order)
    samples = 100000
    steps = 12
    max_num = order - 1

    mean_df = pd.DataFrame()
    std_df = pd.DataFrame()

    means, stds = get_new_prediction(
        df_preds=df_pred,
        input_df=input,
        parameters=parameters,
        samples=samples,
        first=True,
        order=4,
    )
    # print(means)
    mean_df = pd.concat([mean_df, means]).reset_index(drop=True)
    std_df = pd.concat([std_df, stds]).reset_index(drop=True)

    for i in range(steps - 1):
        means, stds = get_new_prediction(
            df_preds=means,
            input_df=input,
            parameters=parameters,
            samples=samples,
            first=False,
            df_stdev=stds,
            order=4,
        )
        mean_df = pd.concat([mean_df, means]).reset_index(drop=True)
        std_df = pd.concat([std_df, stds]).reset_index(drop=True)
    mean_df = mean_df[["city_", "year_month", f"med_housing_{max_num}"]]
    mean_df.insert(
        1, "year", (round(mean_df.year_month - (mean_df.year_month % 1), 0)).astype(int)
    )
    mean_df.insert(2, "month", round(mean_df.year_month % 1 * 100, 0).astype(int))
    mean_df.insert(3, "day", 1)
    mean_df.insert(4, "date", pd.to_datetime(mean_df[["year", "month", "day"]]))

    std_df = std_df[["city_", "year_month", f"med_housing_{max_num}"]]
    std_df.insert(
        1, "year", (round(std_df.year_month - (std_df.year_month % 1), 0)).astype(int)
    )
    std_df.insert(2, "month", round(std_df.year_month % 1 * 100, 0).astype(int))
    std_df.insert(3, "day", 1)
    std_df.insert(4, "date", pd.to_datetime(std_df[["year", "month", "day"]]))

    return mean_df, std_df


def generate_comparative_lineplot(df, mean_df, start_year, max_year, city, order):

    df_line1 = df.query(
        f"year > {start_year-1} & year < {max_year+1} & city == '{city}'"
    )
    df_line1["type"] = "actual"

    df_line2 = mean_df.rename({f"med_housing_{order-1}": "med_housing"}, axis=1)
    df_line2["type"] = "prediction"

    df_line = pd.concat([df_line1, df_line2])
    df_line.day = 1
    df_line["date"] = pd.to_datetime(df_line[["year", "month", "day"]])

    df_line = df_line[["city", "date", "med_housing", "type"]].reset_index(drop=True)

    sns.set(rc={"figure.figsize": (15, 8)})

    sns.lineplot(data=df_line, x="date", y="med_housing", hue="type").set_title(
        f"Meidan Housing Price in Atlanta - Actual v Prediction (VAR order {order})",
        fontsize=20,
    )


def plot_uncertainty(mean_df, std_df, steps):

    xs = []
    ys = []
    dists = []
    for dist in range(steps):
        mean = mean_df.med_housing_3.loc[dist]
        std = std_df.med_housing_3.loc[dist]
        new_xs = np.arange(-4 * std + mean, 4 * std + mean, 100)
        xs.extend(new_xs)
        ys.extend(norm.pdf(new_xs, mean, std))
        dists.extend([dist + 1 for i in range(len(new_xs))])
    plot_df = pd.DataFrame(
        {"median_housing_price": xs, "pdf": ys, "months_in_future": dists}
    )

    sns.set(rc={"figure.figsize": (15, 8)})
    g = sns.FacetGrid(plot_df, col="months_in_future", col_wrap=4)
    g.map(sns.lineplot, "median_housing_price", "pdf")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Distribution of prediction by month")
