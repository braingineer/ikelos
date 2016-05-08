from keras.layers import Dense

class Feeder(object):
	def __init__(self, glove_fp):
		self.glove_fp = glove_fp	

	def run_once(self):
		wleft = lambda w: "#"*2+word
		wright = lambda w: word+"#"*2
		wmiddle = lambda w: "#"+word+"#"
		with open(self.glove_fp) as fp:
			for line in fp.readlines():
	            line = line.replace("\n","").split(" ")
	            word,nums = line[0], [float(x.strip()) for x in line[1:]]
	            word_hash = ["".join(letters) for letters in zip(wleft(word), wmiddle(word), wright(word))]

	def run_forever(self, order=3):
		while True:
