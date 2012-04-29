import math
import copy
#
# Written by Filip Bonnevier
#

# Global variables
classifier_params = [[],[]]
number_correct = 0
number_incorrect = 0
small_float = 1e-100
big_float = 1e100
feature_type = []
# End global variables

def main():
	input_data = []
	while(True):
		try:
			read_input = raw_input()
			if(read_input != ''):
				input_data.append(read_input)
		except EOFError:
			break
	
	if(len(input_data) < 2):
		return

	first_line = input_data[0].split(' ')
	training_set_length = int(first_line[0])
	number_of_features = int(first_line[1])

	training_set = input_data[1:training_set_length+1]
	classify_set_length = input_data[training_set_length+1]
	classify_set = input_data[training_set_length+2:]

	classifier_params[0] = [{} for x in xrange(number_of_features)]
	classifier_params[1] = [{} for x in xrange(number_of_features)]

	training_data = get_params_from_data(training_set, number_of_features, training_set_length)

	train(training_data, number_of_features, training_set_length)

	chosen_features_idx = choose_features(number_of_features)
	number_of_features = len(chosen_features_idx)
	run_classify(classify_set, classify_set_length, number_of_features, chosen_features_idx)

def run_classify(classify_set, classify_set_length, number_of_features, chosen_features_idx):
	for c in classify_set:
		sample = split_data(c)
		posteriors = [0,0]
		for i in xrange(len(classifier_params)):
			p = 1.0
			for idx in chosen_features_idx:
				val = sample['features'][idx]
				if(feature_type[idx] == 'gaussian'):
					p *= calculate_p(val, classifier_params[i][idx]['mean'], classifier_params[i][idx]['variance'])
					if(classifier_params[i][idx]['mean'] == 0):
						print 'zero variance at ' + str(idx)
				if(feature_type[idx] == 'discrete'):
					key = int(val)
					if(i == 0):
						if(key in classifier_params[i][idx]['good_prob_table'].keys()):
							p *= classifier_params[i][idx]['good_prob_table'][int(val)]
						else:
							p *= 0.0
					if(i == 1):
						if(key in classifier_params[i][idx]['bad_prob_table'].keys()):
							p *= classifier_params[i][idx]['bad_prob_table'][int(val)]
						else:
							p *= 0.0
			posteriors[i] = p
		result = sample['id'] + ' '
		if (posteriors[0] > posteriors[1]): # will reject those with same likelihood
			result += '+1'
		else:
			result += '-1'
		print result

# Returns probability for the sample from the PDF which is given by the mean and variance.
def calculate_p(sample, mean, variance):
	if(variance == 0): # Shouldn't happen but just in case
		if(sample == mean):
			return 1
		else:
			return 0
	normalization = 1.0 / math.sqrt(2.0 * math.pi * variance)
	smm_square = (sample-mean)*(sample-mean)
	exponent = (-smm_square / (2.0 * variance))
	x = math.exp(exponent)
	p = normalization * x
	return p

def train(training_data, number_of_features, training_set_length):
	# i is the class (good or bad = 0 or 1)
	for i in xrange(len(training_data)):
		for j in xrange(number_of_features):
			if(feature_type[j] == 'gaussian'):
				feature = {}
				feature['mean'] = calculate_mean(training_data[i][j])
				feature['variance'] = calculate_variance(training_data[i][j], feature['mean'], training_set_length)
				classifier_params[i][j] = feature
	
	# Now train the discrete model. Yes it's a quick hack.
	for j in xrange(number_of_features):
		if(feature_type[j] == 'discrete'):
			good_feature = {}
			bad_feature = {}
			good_data = training_data[0][j]
			bad_data = training_data[1][j]
			temp_joint_list = copy.copy(good_data)
			temp_joint_list.extend(bad_data)
			unique_classes = unique(temp_joint_list)
			if(len(unique_classes) < 2):
				good_feature['reject'] = True
				bad_feature['reject'] = True
			good_feature['n'] = len(unique_classes)
			bad_feature['n'] = len(unique_classes)
			good_feature['p'] = 0.5
			bad_feature['p'] = 0.5
			good_feature['m'] = float(training_set_length) / 1.0
			bad_feature['m'] = float(training_set_length) / 1.0
			good_prob_table = {}
			bad_prob_table = {}

			for uc in unique_classes:
				good_prob = float(good_data.count(uc) + good_feature['m'] * good_feature['p']) / float(good_feature['n'] + good_feature['m'])
				good_prob_table[int(uc)] = good_prob
				bad_prob = float(bad_data.count(uc) + bad_feature['m'] * bad_feature['p']) / float(bad_feature['n'] + bad_feature['m'])
				bad_prob_table[int(uc)] = bad_prob

			good_feature['good_prob_table'] = good_prob_table
			bad_feature['bad_prob_table'] = bad_prob_table
			
			classifier_params[0][j] = good_feature
			classifier_params[1][j] = bad_feature

# Rejects features that have mean/variance == 0 and tries to choose features with large separation
def choose_features(number_of_features):
	
	chosen_features = []
	chosen_features_idx = []
	for j in xrange(number_of_features):
		
		if ('reject' in classifier_params[0][j].keys()):
			#print 'rejected: ' + str(j)
			continue

		if(feature_type[j] == 'gaussian'):
			good_m = classifier_params[0][j]['mean']
			good_v = classifier_params[0][j]['variance']
			bad_m = classifier_params[1][j]['mean']
			bad_v = classifier_params[1][j]['variance']

			if(good_m == 0 and good_v == 0 and bad_m == 0 and bad_v == 0 ):
				continue

			good_s = math.sqrt(good_v)
			bad_s = math.sqrt(bad_v)
			factors = [x * 0.05 for x in range(0, 101)] # start with 5 sigma and work down to 0 sigma
			factors.reverse()
			# This is a simple way to check the overlap between different classes for a feature and 
			# only choose the ones with the least overlap. Only applicable if the number of chosen features are less than the number given.
			# (This will also automatically remove any features that have good_m = bad_m and good_v = bad_v = 0.)
			for factor in factors:
				if(good_m < bad_m):
					if((good_m + factor*good_s) < (bad_m - factor*bad_s)):
						chosen_features.append((factor, j))
						break
				if(good_m > bad_m):
					if((good_m - factor*good_s) > (bad_m + factor*bad_s)):
						chosen_features.append((factor, j))
						break
		
		if(feature_type[j] == 'discrete'):
			chosen_features.append((6, j))

	sorted_features = sorted(chosen_features, key=lambda sort_key:sort_key[0], reverse=True)
	# Choose the x best features (less if there aren't that many features available).
	# The sample data gives a hit rate of 100% already for 8 features but when submitting even 12 was not enough.
	chosen_features = sorted_features[0:25] # 100 is maximum, does not seem to make it run too slow anyways.
	for f,i in chosen_features:
		chosen_features_idx.append(i)
	return chosen_features_idx

def calculate_mean(data):
	mean = math.fsum(data) / float(len(data))
	return mean

def calculate_variance(data, mean, length):
	summ = 0.0
	for value in data:
		summ += (value - mean) * (value - mean) #same as math.pow((value-mean),2) but faster
	try:
		variance = summ / float(length - 1)
	except:
		print 'Length of the training set is 1 or less. Must be larger than 1.'
	return variance

# Returns data on the form [class][feature][sample] where class is 0 for 'good' and 1 for 'bad'.
def get_params_from_data(training_set, number_of_features, training_set_length):
	bad_data = [[] for j in xrange(number_of_features)]
	good_data = [[] for j in xrange(number_of_features)]
	data = [good_data, bad_data]
	
	# Start by splitting the data up and convert it to floats. Also check if there are any '(-)inf' sneaked in anywhere.
	for i, line in enumerate(training_set):
		split = line.split(' ')
		good_or_bad_class = split[1]
		features = split[2:]
		if (good_or_bad_class == '+1'):
			data_to_append = good_data
		else:
			data_to_append = bad_data
		for j in xrange(number_of_features):
			s = features[j]
			s_split = s.split(':')
			num = float(s_split[1])
			if(num == float('inf')):
				num = big_float
			if (num == float('-inf')):
				num = small_float
			data_to_append[j].append(num)
	
	# There are discrete distributions in the data set which not should be handled like Gaussians
	# Start by assuming that all features are gaussian and then check which ones are discrete
	global feature_type
	feature_type = ['gaussian' for j in xrange(number_of_features)]
	for cls in data:
		for i, feature in enumerate(cls):
			unique_values = []
			len_unique_values = float(len(unique(feature)))
			len_feature = float(len(feature))
			if((len_unique_values / len_feature) < 0.6): # Ratio of distinct values will be less than 1 at least.
				feature_type[i] = 'discrete'             # I chose 0.6 to cater for small data sets (like the test set)
				
	return data

# Returns a unique list from a given list
def unique(input_list):
	keys = {}
	for item in input_list:
		keys[item] = 1
	return keys.keys()

# Splits a line of sample data (and training data if needed).
def split_data(line):
	split = line.split(' ')
	data = {}
	data['id'] = split[0]
	
	if(split[1].startswith('+') or split[1].startswith('-')):
		data['result'] = split[1]
		data['features'] = split[2:]
	else:
		data['features'] = split[1:]
	
	splitted_features = []
	for f in data['features']:
		f_split = f.split(':')
		num = float(f_split[1])
		if(num == float('inf')):
			num = big_float
		if (num == float('-inf')):
			num = small_float
		splitted_features.append(num)
	data['features'] = splitted_features

	return data

if __name__ == '__main__':
	main()