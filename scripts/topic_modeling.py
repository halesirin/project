import argparse
import gzip
import re
import json
from pprint import pprint
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", dest="input", help="Input file")
	parser.add_argument("-o", "--output", dest="output", help="Output file")
	args = parser.parse_args()

	word_to_column = {}

	with gzip.open(args.input, "rt") as ifd:
		for line in ifd:
			obj = json.loads(line)
			text = obj["content"].lower().strip()
			tokens = re.split(r"\W+", text)
			for token in tokens:
				if token not in word_to_column:
					word_to_column[token] = len(word_to_column) #how did we add the word itself?
	

	counts = {}
	with gzip.open(args.input, "rt") as ifd:
		#for doc_num, line in enumerate(ifd):
		for line in ifd:

			obj = json.loads(line)
			text = obj["content"].lower().strip()
			tokens = re.split(r"\W+", text)
			ntoks =  len(tokens)
			doclen = ntoks // 100
			for subdoc in range(ntoks // doclen):
				doc_num = len(counts)
				subtoks = tokens[subdoc * doclen : (subdoc + 1) * doclen]
				counts[doc_num] = {}
				for token in subtoks:
					word_num = word_to_column[token]
					counts[doc_num][word_num] = counts[doc_num].get(word_num, 0) + 1 # how did "get" work here? 
				
				
				#if word_num in counts[doc_num]:
				#	counts[doc_num][word_num] = counts[doc_num][word_num] + 1
				#else:
				#	counts[doc_num][word_num] = 1
				
	# Train LDA model.
	from gensim.models import LdaModel

	# Set training parameters.
	num_topics = 100
	chunksize = 2000
	passes = 20
	iterations = 400
	eval_every = None  # Don't evaluate model perplexity, takes too much time.

	id2word = {v : k for k, v in word_to_column.items()}
	corpus = [[(word, count) for word, count in doc.items()] for doc in counts.values()]


	model = LdaModel(
    	corpus=corpus,
    	id2word=id2word,
    	chunksize=chunksize,
    	alpha='auto',
    	eta='auto',
    	iterations=iterations,
    	num_topics=num_topics,
    	passes=passes,
    	eval_every=eval_every
	)

	top_topics = model.top_topics(corpus) #, num_words=20)

	# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
	avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
	print('Average topic coherence: %.4f.' % avg_topic_coherence)

	from pprint import pprint
	pprint(top_topics)
