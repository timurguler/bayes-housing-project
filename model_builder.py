import pymc3 as pm
def build_model(input_df, sig=1000000):
	with pm.Model() as model_comb:
		# med_housing
		med_housing_0_med_housing = pm.Normal(name='med_housing_0_med_housing', mu=0, sigma=sig)
		med_housing_1_med_housing = pm.Normal(name='med_housing_1_med_housing', mu=0, sigma=sig)
		med_housing_2_med_housing = pm.Normal(name='med_housing_2_med_housing', mu=0, sigma=sig)
		med_housing_3_med_housing = pm.Normal(name='med_housing_3_med_housing', mu=0, sigma=sig)
		unemployment_0_med_housing = pm.Normal(name='unemployment_0_med_housing', mu=0, sigma=sig)
		unemployment_1_med_housing = pm.Normal(name='unemployment_1_med_housing', mu=0, sigma=sig)
		unemployment_2_med_housing = pm.Normal(name='unemployment_2_med_housing', mu=0, sigma=sig)
		unemployment_3_med_housing = pm.Normal(name='unemployment_3_med_housing', mu=0, sigma=sig)
		housing_listings_0_med_housing = pm.Normal(name='housing_listings_0_med_housing', mu=0, sigma=sig)
		housing_listings_1_med_housing = pm.Normal(name='housing_listings_1_med_housing', mu=0, sigma=sig)
		housing_listings_2_med_housing = pm.Normal(name='housing_listings_2_med_housing', mu=0, sigma=sig)
		housing_listings_3_med_housing = pm.Normal(name='housing_listings_3_med_housing', mu=0, sigma=sig)
		population_0_med_housing = pm.Normal(name='population_0_med_housing', mu=0, sigma=sig)
		population_1_med_housing = pm.Normal(name='population_1_med_housing', mu=0, sigma=sig)
		population_2_med_housing = pm.Normal(name='population_2_med_housing', mu=0, sigma=sig)
		population_3_med_housing = pm.Normal(name='population_3_med_housing', mu=0, sigma=sig)
		income_0_med_housing = pm.Normal(name='income_0_med_housing', mu=0, sigma=sig)
		income_1_med_housing = pm.Normal(name='income_1_med_housing', mu=0, sigma=sig)
		income_2_med_housing = pm.Normal(name='income_2_med_housing', mu=0, sigma=sig)
		income_3_med_housing = pm.Normal(name='income_3_med_housing', mu=0, sigma=sig)

		theta_med_housing = (
			med_housing_0_med_housing*input_df.med_housing_0+
			med_housing_1_med_housing*input_df.med_housing_1+
			med_housing_2_med_housing*input_df.med_housing_2+
			med_housing_3_med_housing*input_df.med_housing_3+
			unemployment_0_med_housing*input_df.unemployment_0+
			unemployment_1_med_housing*input_df.unemployment_1+
			unemployment_2_med_housing*input_df.unemployment_2+
			unemployment_3_med_housing*input_df.unemployment_3+
			housing_listings_0_med_housing*input_df.housing_listings_0+
			housing_listings_1_med_housing*input_df.housing_listings_1+
			housing_listings_2_med_housing*input_df.housing_listings_2+
			housing_listings_3_med_housing*input_df.housing_listings_3+
			population_0_med_housing*input_df.population_0+
			population_1_med_housing*input_df.population_1+
			population_2_med_housing*input_df.population_2+
			population_3_med_housing*input_df.population_3+
			income_0_med_housing*input_df.income_0+
			income_1_med_housing*input_df.income_1+
			income_2_med_housing*input_df.income_2+
			income_3_med_housing*input_df.income_3
		)
		med_housing = pm.Normal('med_housing', theta_med_housing, sd=200, observed=input_df.med_housing)
	
		# unemployment
		med_housing_0_unemployment = pm.Normal(name='med_housing_0_unemployment', mu=0, sigma=sig)
		med_housing_1_unemployment = pm.Normal(name='med_housing_1_unemployment', mu=0, sigma=sig)
		med_housing_2_unemployment = pm.Normal(name='med_housing_2_unemployment', mu=0, sigma=sig)
		med_housing_3_unemployment = pm.Normal(name='med_housing_3_unemployment', mu=0, sigma=sig)
		unemployment_0_unemployment = pm.Normal(name='unemployment_0_unemployment', mu=0, sigma=sig)
		unemployment_1_unemployment = pm.Normal(name='unemployment_1_unemployment', mu=0, sigma=sig)
		unemployment_2_unemployment = pm.Normal(name='unemployment_2_unemployment', mu=0, sigma=sig)
		unemployment_3_unemployment = pm.Normal(name='unemployment_3_unemployment', mu=0, sigma=sig)
		housing_listings_0_unemployment = pm.Normal(name='housing_listings_0_unemployment', mu=0, sigma=sig)
		housing_listings_1_unemployment = pm.Normal(name='housing_listings_1_unemployment', mu=0, sigma=sig)
		housing_listings_2_unemployment = pm.Normal(name='housing_listings_2_unemployment', mu=0, sigma=sig)
		housing_listings_3_unemployment = pm.Normal(name='housing_listings_3_unemployment', mu=0, sigma=sig)
		population_0_unemployment = pm.Normal(name='population_0_unemployment', mu=0, sigma=sig)
		population_1_unemployment = pm.Normal(name='population_1_unemployment', mu=0, sigma=sig)
		population_2_unemployment = pm.Normal(name='population_2_unemployment', mu=0, sigma=sig)
		population_3_unemployment = pm.Normal(name='population_3_unemployment', mu=0, sigma=sig)
		income_0_unemployment = pm.Normal(name='income_0_unemployment', mu=0, sigma=sig)
		income_1_unemployment = pm.Normal(name='income_1_unemployment', mu=0, sigma=sig)
		income_2_unemployment = pm.Normal(name='income_2_unemployment', mu=0, sigma=sig)
		income_3_unemployment = pm.Normal(name='income_3_unemployment', mu=0, sigma=sig)

		theta_unemployment = (
			med_housing_0_unemployment*input_df.med_housing_0+
			med_housing_1_unemployment*input_df.med_housing_1+
			med_housing_2_unemployment*input_df.med_housing_2+
			med_housing_3_unemployment*input_df.med_housing_3+
			unemployment_0_unemployment*input_df.unemployment_0+
			unemployment_1_unemployment*input_df.unemployment_1+
			unemployment_2_unemployment*input_df.unemployment_2+
			unemployment_3_unemployment*input_df.unemployment_3+
			housing_listings_0_unemployment*input_df.housing_listings_0+
			housing_listings_1_unemployment*input_df.housing_listings_1+
			housing_listings_2_unemployment*input_df.housing_listings_2+
			housing_listings_3_unemployment*input_df.housing_listings_3+
			population_0_unemployment*input_df.population_0+
			population_1_unemployment*input_df.population_1+
			population_2_unemployment*input_df.population_2+
			population_3_unemployment*input_df.population_3+
			income_0_unemployment*input_df.income_0+
			income_1_unemployment*input_df.income_1+
			income_2_unemployment*input_df.income_2+
			income_3_unemployment*input_df.income_3
		)
		unemployment = pm.Normal('unemployment', theta_unemployment, sd=200, observed=input_df.unemployment)
	
		# housing_listings
		med_housing_0_housing_listings = pm.Normal(name='med_housing_0_housing_listings', mu=0, sigma=sig)
		med_housing_1_housing_listings = pm.Normal(name='med_housing_1_housing_listings', mu=0, sigma=sig)
		med_housing_2_housing_listings = pm.Normal(name='med_housing_2_housing_listings', mu=0, sigma=sig)
		med_housing_3_housing_listings = pm.Normal(name='med_housing_3_housing_listings', mu=0, sigma=sig)
		unemployment_0_housing_listings = pm.Normal(name='unemployment_0_housing_listings', mu=0, sigma=sig)
		unemployment_1_housing_listings = pm.Normal(name='unemployment_1_housing_listings', mu=0, sigma=sig)
		unemployment_2_housing_listings = pm.Normal(name='unemployment_2_housing_listings', mu=0, sigma=sig)
		unemployment_3_housing_listings = pm.Normal(name='unemployment_3_housing_listings', mu=0, sigma=sig)
		housing_listings_0_housing_listings = pm.Normal(name='housing_listings_0_housing_listings', mu=0, sigma=sig)
		housing_listings_1_housing_listings = pm.Normal(name='housing_listings_1_housing_listings', mu=0, sigma=sig)
		housing_listings_2_housing_listings = pm.Normal(name='housing_listings_2_housing_listings', mu=0, sigma=sig)
		housing_listings_3_housing_listings = pm.Normal(name='housing_listings_3_housing_listings', mu=0, sigma=sig)
		population_0_housing_listings = pm.Normal(name='population_0_housing_listings', mu=0, sigma=sig)
		population_1_housing_listings = pm.Normal(name='population_1_housing_listings', mu=0, sigma=sig)
		population_2_housing_listings = pm.Normal(name='population_2_housing_listings', mu=0, sigma=sig)
		population_3_housing_listings = pm.Normal(name='population_3_housing_listings', mu=0, sigma=sig)
		income_0_housing_listings = pm.Normal(name='income_0_housing_listings', mu=0, sigma=sig)
		income_1_housing_listings = pm.Normal(name='income_1_housing_listings', mu=0, sigma=sig)
		income_2_housing_listings = pm.Normal(name='income_2_housing_listings', mu=0, sigma=sig)
		income_3_housing_listings = pm.Normal(name='income_3_housing_listings', mu=0, sigma=sig)

		theta_housing_listings = (
			med_housing_0_housing_listings*input_df.med_housing_0+
			med_housing_1_housing_listings*input_df.med_housing_1+
			med_housing_2_housing_listings*input_df.med_housing_2+
			med_housing_3_housing_listings*input_df.med_housing_3+
			unemployment_0_housing_listings*input_df.unemployment_0+
			unemployment_1_housing_listings*input_df.unemployment_1+
			unemployment_2_housing_listings*input_df.unemployment_2+
			unemployment_3_housing_listings*input_df.unemployment_3+
			housing_listings_0_housing_listings*input_df.housing_listings_0+
			housing_listings_1_housing_listings*input_df.housing_listings_1+
			housing_listings_2_housing_listings*input_df.housing_listings_2+
			housing_listings_3_housing_listings*input_df.housing_listings_3+
			population_0_housing_listings*input_df.population_0+
			population_1_housing_listings*input_df.population_1+
			population_2_housing_listings*input_df.population_2+
			population_3_housing_listings*input_df.population_3+
			income_0_housing_listings*input_df.income_0+
			income_1_housing_listings*input_df.income_1+
			income_2_housing_listings*input_df.income_2+
			income_3_housing_listings*input_df.income_3
		)
		housing_listings = pm.Normal('housing_listings', theta_housing_listings, sd=200, observed=input_df.housing_listings)
	
		# population
		med_housing_0_population = pm.Normal(name='med_housing_0_population', mu=0, sigma=sig)
		med_housing_1_population = pm.Normal(name='med_housing_1_population', mu=0, sigma=sig)
		med_housing_2_population = pm.Normal(name='med_housing_2_population', mu=0, sigma=sig)
		med_housing_3_population = pm.Normal(name='med_housing_3_population', mu=0, sigma=sig)
		unemployment_0_population = pm.Normal(name='unemployment_0_population', mu=0, sigma=sig)
		unemployment_1_population = pm.Normal(name='unemployment_1_population', mu=0, sigma=sig)
		unemployment_2_population = pm.Normal(name='unemployment_2_population', mu=0, sigma=sig)
		unemployment_3_population = pm.Normal(name='unemployment_3_population', mu=0, sigma=sig)
		housing_listings_0_population = pm.Normal(name='housing_listings_0_population', mu=0, sigma=sig)
		housing_listings_1_population = pm.Normal(name='housing_listings_1_population', mu=0, sigma=sig)
		housing_listings_2_population = pm.Normal(name='housing_listings_2_population', mu=0, sigma=sig)
		housing_listings_3_population = pm.Normal(name='housing_listings_3_population', mu=0, sigma=sig)
		population_0_population = pm.Normal(name='population_0_population', mu=0, sigma=sig)
		population_1_population = pm.Normal(name='population_1_population', mu=0, sigma=sig)
		population_2_population = pm.Normal(name='population_2_population', mu=0, sigma=sig)
		population_3_population = pm.Normal(name='population_3_population', mu=0, sigma=sig)
		income_0_population = pm.Normal(name='income_0_population', mu=0, sigma=sig)
		income_1_population = pm.Normal(name='income_1_population', mu=0, sigma=sig)
		income_2_population = pm.Normal(name='income_2_population', mu=0, sigma=sig)
		income_3_population = pm.Normal(name='income_3_population', mu=0, sigma=sig)

		theta_population = (
			med_housing_0_population*input_df.med_housing_0+
			med_housing_1_population*input_df.med_housing_1+
			med_housing_2_population*input_df.med_housing_2+
			med_housing_3_population*input_df.med_housing_3+
			unemployment_0_population*input_df.unemployment_0+
			unemployment_1_population*input_df.unemployment_1+
			unemployment_2_population*input_df.unemployment_2+
			unemployment_3_population*input_df.unemployment_3+
			housing_listings_0_population*input_df.housing_listings_0+
			housing_listings_1_population*input_df.housing_listings_1+
			housing_listings_2_population*input_df.housing_listings_2+
			housing_listings_3_population*input_df.housing_listings_3+
			population_0_population*input_df.population_0+
			population_1_population*input_df.population_1+
			population_2_population*input_df.population_2+
			population_3_population*input_df.population_3+
			income_0_population*input_df.income_0+
			income_1_population*input_df.income_1+
			income_2_population*input_df.income_2+
			income_3_population*input_df.income_3
		)
		population = pm.Normal('population', theta_population, sd=200, observed=input_df.population)
	
		# income
		med_housing_0_income = pm.Normal(name='med_housing_0_income', mu=0, sigma=sig)
		med_housing_1_income = pm.Normal(name='med_housing_1_income', mu=0, sigma=sig)
		med_housing_2_income = pm.Normal(name='med_housing_2_income', mu=0, sigma=sig)
		med_housing_3_income = pm.Normal(name='med_housing_3_income', mu=0, sigma=sig)
		unemployment_0_income = pm.Normal(name='unemployment_0_income', mu=0, sigma=sig)
		unemployment_1_income = pm.Normal(name='unemployment_1_income', mu=0, sigma=sig)
		unemployment_2_income = pm.Normal(name='unemployment_2_income', mu=0, sigma=sig)
		unemployment_3_income = pm.Normal(name='unemployment_3_income', mu=0, sigma=sig)
		housing_listings_0_income = pm.Normal(name='housing_listings_0_income', mu=0, sigma=sig)
		housing_listings_1_income = pm.Normal(name='housing_listings_1_income', mu=0, sigma=sig)
		housing_listings_2_income = pm.Normal(name='housing_listings_2_income', mu=0, sigma=sig)
		housing_listings_3_income = pm.Normal(name='housing_listings_3_income', mu=0, sigma=sig)
		population_0_income = pm.Normal(name='population_0_income', mu=0, sigma=sig)
		population_1_income = pm.Normal(name='population_1_income', mu=0, sigma=sig)
		population_2_income = pm.Normal(name='population_2_income', mu=0, sigma=sig)
		population_3_income = pm.Normal(name='population_3_income', mu=0, sigma=sig)
		income_0_income = pm.Normal(name='income_0_income', mu=0, sigma=sig)
		income_1_income = pm.Normal(name='income_1_income', mu=0, sigma=sig)
		income_2_income = pm.Normal(name='income_2_income', mu=0, sigma=sig)
		income_3_income = pm.Normal(name='income_3_income', mu=0, sigma=sig)

		theta_income = (
			med_housing_0_income*input_df.med_housing_0+
			med_housing_1_income*input_df.med_housing_1+
			med_housing_2_income*input_df.med_housing_2+
			med_housing_3_income*input_df.med_housing_3+
			unemployment_0_income*input_df.unemployment_0+
			unemployment_1_income*input_df.unemployment_1+
			unemployment_2_income*input_df.unemployment_2+
			unemployment_3_income*input_df.unemployment_3+
			housing_listings_0_income*input_df.housing_listings_0+
			housing_listings_1_income*input_df.housing_listings_1+
			housing_listings_2_income*input_df.housing_listings_2+
			housing_listings_3_income*input_df.housing_listings_3+
			population_0_income*input_df.population_0+
			population_1_income*input_df.population_1+
			population_2_income*input_df.population_2+
			population_3_income*input_df.population_3+
			income_0_income*input_df.income_0+
			income_1_income*input_df.income_1+
			income_2_income*input_df.income_2+
			income_3_income*input_df.income_3
		)
		income = pm.Normal('income', theta_income, sd=200, observed=input_df.income)
	
	return model_comb