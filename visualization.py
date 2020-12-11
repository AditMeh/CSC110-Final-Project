from generate_dictionary import compute_class_word_frequency_dicts
from datareader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

FILEPATH = "twitter_sentiment_data.csv"
loader = DataLoader(FILEPATH)
train_x, train_y = loader.prepare_data()
freq_dict = compute_class_word_frequency_dicts(train_x, train_y)

thing = sorted(freq_dict[-1].items(), key=lambda x: x[1])

thing_1 = [thing[-i][1] for i in range(1, 20)]
thing_2 = [thing[-i][0] for i in range(1, 20)]


y_pos = np.arange(len(thing_1))

plt.bar(y_pos, thing_1, align='center', alpha=0.5)
plt.xticks(y_pos, thing_2, fontsize = 5)
plt.ylabel('Usage')
plt.title('Programming language usage')

plt.show()