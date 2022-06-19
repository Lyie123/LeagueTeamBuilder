import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


class TimeScaler(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		for column in X.columns:
			X[column] = X[column]/X['time_played']
		return X


data = pd.read_csv('.\data\participants.csv')
data = data.dropna(subset=['team_position'])

X_train, X_test, y_train, y_test = train_test_split(
	data.loc[:, data.columns!='team_position'], data['team_position'], test_size=0.2, random_state=42
)

num_attribs = [
	'champ_experience',
	'magic_damage_dealt_to_champions',
	'physical_damage_dealt_to_champions',
	'time_c_cing_others',
	'total_heals_on_teammates',
	'neutral_minions_killed',
	'total_minions_killed',
	'damage_dealt_to_buildings',
	'damage_dealt_to_objectives',
	'damage_dealt_to_turrets',
	'damage_self_mitigated',
	'deaths',
	'kills',
	'assists',
	'vision_score',
	'detector_wards_placed',
	'gold_earned',
	'time_played'
]

cat_attribs = [
	'champion_name',
	'summoner1_id',
	'summoner2_id',
	#'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6',
]

categories = [
	sorted(data['champion_name'].unique()),
	sorted(pd.concat([data['summoner1_id'], data['summoner2_id']]).unique()),
	sorted(pd.concat([data['summoner1_id'], data['summoner2_id']]).unique()),
]

num_pipeline = Pipeline([
	('time_scaler', TimeScaler()),
	('std_scaler', StandardScaler()),
	('min_max_scaler', MinMaxScaler())
])

full_pipeline = ColumnTransformer([
	('num', num_pipeline, num_attribs),
	('cat', OneHotEncoder(categories=categories), cat_attribs),
])


X_train_prepared = full_pipeline.fit_transform(X_train)
sgd_clf = RandomForestClassifier(random_state=44)

score = cross_val_score(sgd_clf, X_train_prepared, y_train.values.ravel(), cv=3, scoring='accuracy')
print(score)

sgd_clf = sgd_clf.fit(X_train_prepared, y_train.values.ravel())

disp = ConfusionMatrixDisplay.from_estimator(sgd_clf, full_pipeline.fit_transform(X_test), y_test.values.ravel(), normalize='true')

plt.show()