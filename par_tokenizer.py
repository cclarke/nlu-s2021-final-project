import spacy
nlp = spacy.load('en_core_web_sm')

def get_sentences(paragraph):
	result = []
	try: 
		doc = nlp(paragraph)
		for sentence in doc.sents:
			result.append(sentence)
	except Exception:
		print("This paragraph could not be converted", paragraph)
	return result