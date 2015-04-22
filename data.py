

import types


class DataSet(object):
	
	def __init__(self, dataclass):
		self.dataclass = dataclass
	
	def load_csv(self, file_path):
		pass
	
	def load_pandas(self, dataframe):
		pass
	
	def load_sql(self, CON, table):
		pass

	
class Data(object):
	
	def __init__(self):
		pass