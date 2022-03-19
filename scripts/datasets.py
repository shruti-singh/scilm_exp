class Datasets:
	dataset_holder = {'ACL': {'name': 'ACL', 'loc': 'datasets/ACL/'}, 'ICLR': {'name': 'ICLR', 'loc': 'datasets/ICLR/'},
					  'ECON': {'name': 'ECON', 'loc': 'datasets/ECON/'},
					  'ArxivHEPTH': {'name': 'ArxivHEPTH', 'loc': 'datasets/ArxivHEPTH/'},
					  'ArxivQBIONC': {'name': 'ArxivQBIONC', 'loc': 'datasets/ArxivQBIONC/'},
					  'ArxivMATHAT': {'name': 'ArxivMATHAT', 'loc': 'datasets/ArxivMATHAT/'},
					  'ArxivCSSY': {'name': 'ArxivCSSY', 'loc': 'datasets/ArxivCSSY/'}}

	@staticmethod
	def add_dataset(name, loc):
		Datasets.dataset_holder[name] = {'name': name, 'loc': loc}

	@staticmethod
	def delete_dataset(name):
		del Datasets.dataset_holder[name]
