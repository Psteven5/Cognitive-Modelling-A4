import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pyddm
import pyddm.plot

# conditional drift rate based on the given instruction
class DriftInstruction(pyddm.DriftConstant):
	name = "Drift according to the given instruction"
	required_conditions = ["instruction"]
	required_parameters = ["drift1", "drift2"]
	
	def get_drift(self, conditions, **kwargs):
		if "speed" == conditions["instruction"]:
			return self.drift1
		return self.drift2

# conditional threshold based on the given instruction
class BoundInstruction(pyddm.BoundConstant):
	name = "Bound according to the given instruction"
	required_conditions = ["instruction"]
	required_parameters = ["bound1", "bound2", "tau"]
	
	def get_bound(self, t, conditions, **kwargs):
		weight = np.exp(-t * self.tau)
		if "speed" == conditions["instruction"]:
			return weight * self.bound1
		return weight * self.bound2

# conditional bias on top of the drift rate based on the given instruction
class DriftBiasInstruction(pyddm.DriftConstant):
	name = "Drift with a bias according to the choice and given instruction"
	required_conditions = ["instruction", 'R']
	required_parameters = ["drift", "bias1", "bias2"]
	def get_drift(self, conditions, **kwargs):
		side = 1 if "left" == conditions['R'] else -1
		if "speed" == conditions["instruction"]:
			return self.drift + side * self.bias1
		return self.drift + side * self.bias2

# plots the RT distribution of a sample
def plot_sample(title, sample, dt=0.005):
	bins = np.arange(start=min(sample), stop = max(sample) + dt, step=dt)
	plt.figure(figsize=(10, 3))
	plt.hist(x=sample.choice_upper, bins=bins, color="cornflowerblue", alpha=0.75)
	plt.hist(x=sample.choice_lower, bins=bins, color="lightcoral",     alpha=0.75)
	plt.xlabel("Reaction time (s)")
	plt.ylabel("Frequency")
	plt.legend(["Correct", "Error"])
	plt.title(title)
	plt.show()

# makes a boxplot with t-test information of the two conditions
def condition_ttest(title, ylabel, col, df):
	group1 = df["speed"    == df["instruction"]][col]
	group2 = df["accuracy" == df["instruction"]][col]
	max_y = max(group1.max(), group2.max())
	min_y = min(group1.min(), group2.min())
	mean_group1 = round(group1.mean(), 2)
	mean_group2 = round(group2.mean(), 2)
	t_value, p_value = stats.ttest_rel(group1, group2)
	grouped = df.copy(deep=True)
	grouped["instruction"] = ["Condition 1" if "speed" == i else "Condition 2" for i in df["instruction"]]
	grouped = grouped.groupby("instruction")[col]
	plt.boxplot([group.values for _, group in grouped], labels=grouped.groups.keys(), medianprops={"color": "black"})
	plt.text(0.55, 0.975 * (max_y - min_y) + min_y, f"t-value: {round(t_value, 2)}")
	plt.text(0.55, 0.925 * (max_y - min_y) + min_y, f"p-value: {round(p_value, 5)}")
	plt.ylabel(ylabel)
	plt.text(1.1, mean_group1, f"mean: {mean_group1}")
	plt.text(2.1, mean_group2, f"mean: {mean_group2}")
	plt.title(title)
	plt.show()

if "__main__" == __name__:
	# to enable the generation of only a selection of figures
	figure_amount = 9
	generate_figures = np.asarray([
		'y' == input(f"Generate figure {i + 1}? [y/n] > ").lower() for i in range(figure_amount)])

	# read tsv
	df = pd.read_csv("dataset-14.tsv", sep='\t')

	# dt for simulations
	dt = 0.0005

	# we have to make a correct column for pyddm
	df["correct"] = (df['R'] == df['S']).astype(np.uint8)

	# plot with outliers, just for the whole picture of the data
	if generate_figures[0]:
		plot_sample("RT distribution", pyddm.Sample.from_pandas_dataframe(df, rt_column_name="rt", correct_column_name="correct"))

	# first we remove the outliers
	# the 5th percentile is the most accurate fit in the case of this dataset
	# (because of the wide range of reaction times caused by the speed-accuracy tradeoff)
	df_ = df[(df["rt"].quantile(0.05) <= df["rt"]) &
		 (df["rt"] < df["rt"].quantile(0.95))]

	sample = pyddm.Sample.from_pandas_dataframe(df_, rt_column_name="rt", correct_column_name="correct")
	T_dur = max(sample) + dt

	# plot RT-distribution with removed outliers
	if generate_figures[1]:
		plot_sample("RT distribution, outliers removed", sample)
	
	# plot the distributions for speed and accuracy separately for good measure
	if generate_figures[2]:
		plot_sample("RT distribution w/ condition 1, outliers removed", pyddm.Sample.from_pandas_dataframe(df_["speed"    == df_["instruction"]], rt_column_name="rt", correct_column_name="correct"))
	if generate_figures[3]:
		plot_sample("RT distribution w/ condition 2, outliers removed", pyddm.Sample.from_pandas_dataframe(df_["accuracy" == df_["instruction"]], rt_column_name="rt", correct_column_name="correct"))

	# t-test for the reaction times between conditions
	if generate_figures[4]:
		condition_ttest("Student t-test between condition 1 and 2 (reaction times)", "Reaction time (s)", "rt", df)
	
	# t-test for the mean accuracy between conditions
	if generate_figures[5]:
		tmp = pd.DataFrame({
			"subject": [i + 1 for i in range(12) for _ in range(2)],
			"instruction":   itertools.chain(*[["speed", "accuracy"] for _ in range(12)]),
			"mean_accuracy": itertools.chain(*[[np.mean(df_[("speed"    == df_["instruction"]) & (i + 1 == df_["subjects"])]["correct"]),
			                                    np.mean(df_[("accuracy" == df_["instruction"]) & (i + 1 == df_["subjects"])]["correct"])] for i in range(12)])})
		condition_ttest("Student t-test between condition 1 and 2 (mean accuracy)", "Mean accuracy", "mean_accuracy", tmp)

	# model A
	if generate_figures[6]:
		model = pyddm.Model(drift=DriftInstruction(drift1=pyddm.Fittable(minval=-20.0, maxval=20.0), drift2=pyddm.Fittable(minval=-20.0, maxval=20.0)),
				    noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=0.1, maxval=1.0)),
				    bound=pyddm.BoundCollapsingExponential(B=pyddm.Fittable(minval=0.1, maxval=2.0), tau=pyddm.Fittable(minval=0.01, maxval=3.0)),
				    IC=pyddm.ICRange(sz=pyddm.Fittable(minval=0, maxval=.9)),
				    overlay=pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0.0, maxval=0.5)),
				    T_dur=T_dur, dt=dt)
		pyddm.fit_adjust_model(model=model, sample=sample, lossfunction=pyddm.LossRobustLikelihood, verbose=False)
		drift1, drift2, noise, B, tau, sz, nondectime = model.get_model_parameters()
		print(f"""
Model A (conditional drift rate)   loss  {model.get_fit_result().value()}
    drift1  {drift1.real}
    drift2  {drift2.real}
     noise  {noise.real}
         B  {B.real}
       tau  {tau.real}
        sz  {sz.real}
nondectime  {nondectime.real}
~
""")
		pyddm.plot.model_gui(model, sample)

	# model B
	if generate_figures[7]:
		model = pyddm.Model(drift=pyddm.DriftConstant(drift=pyddm.Fittable(minval=-20.0, maxval=20.0)),
				    noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=0.1, maxval=1.0)),
				    bound=BoundInstruction(bound1=pyddm.Fittable(minval=0.1, maxval=2.0), bound2=pyddm.Fittable(minval=0.1, maxval=2.0), tau=pyddm.Fittable(minval=0.01, maxval=3.0)),
				    IC=pyddm.ICRange(sz=pyddm.Fittable(minval=0, maxval=.9)),
				    overlay=pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0.0, maxval=0.5)),
				    T_dur=T_dur, dt=dt)
		pyddm.fit_adjust_model(model=model, sample=sample, lossfunction=pyddm.LossRobustLikelihood, verbose=False)
		drift, noise, bound1, bound2, tau, sz, nondectime = model.get_model_parameters()
		print(f"""
Model B (conditional threshold)   loss  {model.get_fit_result().value()}
     drift  {drift.real}
     noise  {noise.real}
    bound1  {bound1.real}
    bound2  {bound2.real}
       tau  {tau.real}
        sz  {sz.real}
nondectime  {nondectime.real}
~
""")
		pyddm.plot.model_gui(model, sample)

	# model C
	if generate_figures[8]:
		model = pyddm.Model(drift=DriftBiasInstruction(drift=pyddm.Fittable(minval=-20.0, maxval=20.0), bias1=pyddm.Fittable(minval=-0.8, maxval=0.8), bias2=pyddm.Fittable(minval=-0.8, maxval=0.8)),
				    noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=0.1, maxval=1.0)),
				    bound=pyddm.BoundCollapsingExponential(B=pyddm.Fittable(minval=0.1, maxval=2.0), tau=pyddm.Fittable(minval=0.01, maxval=3.0)),
				    IC=pyddm.ICRange(sz=pyddm.Fittable(minval=0, maxval=.9)),
				    overlay=pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0.0, maxval=0.5)),
				    T_dur=T_dur, dt=dt)
		pyddm.fit_adjust_model(model=model, sample=sample, lossfunction=pyddm.LossRobustLikelihood, verbose=False)
		drift, bias1, bias2, noise, B, tau, sz, nondectime = model.get_model_parameters()
		print(f"""
Model C (conditional response bias)   loss {model.get_fit_result().value()}
     drift  {drift.real}
     bias1  {bias1.real}
     bias2  {bias2.real}
     noise  {noise.real}
         B  {B.real}
       tau  {tau.real}
        sz  {sz.real}
nondectime  {nondectime.real}
~
""")
		pyddm.plot.model_gui(model, sample)
