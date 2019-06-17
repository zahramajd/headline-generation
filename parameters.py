top_freq_word_to_use = 40000
embedding_dimension = 300
max_len_head = 25
max_len_desc = 50
max_length = max_len_head + max_len_desc
rnn_layers = 4
rnn_size = 600

activation_rnn_size = 50
learning_rate = 1e-4

min_head_line_gen = 10
dont_repeat_word_in_last = 5

train_val_percent = 0.9
test_size = 100


dataLen = 10000
train_size = int((dataLen - test_size)* train_val_percent)
val_size = dataLen - train_size - test_size


headlines_path = "HEADLINE_Generation/sample.headline.data"
contents_path = "HEADLINE_Generation/sample.content.data"