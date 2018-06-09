Wiki dataset version 2.0, we should upgrade to 3.0 soon.
The data:
	paired_array_normal_simple.npy

	is an array (not a matrix, data is not padded).

	the first dimension is normal and simple, i. e.

	data = np.load('paired_array_normal_simple.npy')
	data[0] -> gives you all sequences in "normal" language
	data[1] -> gives you all sequences in "simple" language
	sequences with the same index have the same content, i. e. 
	data[1][i] is the simplification of data[0][i] for all i

alphabets:
	simple_dict.pkl
	and
	normal_dict.pkl

	map the sequence indices to words
	The vocabulary is only in lower case, to have a smaller overall vocabulary (among other reasons)

