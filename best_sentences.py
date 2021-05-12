from tqdm import tqdm 


list_results = ["Cameron-BERT-eec-emotion-00:40:36-conv_ai_3-train-eval.out", "Cameron-BERT-eec-emotion-00:41:55-air_dialogue-train-eval.out", 
	"Cameron-BERT-eec-emotion-00:53:13-empathetic_dialogues-train-eval.out", "Cameron-BERT-eec-emotion-01:24:02-ted_talks_iwslt-train-eval.out", 
	"Cameron-BERT-eec-emotion-09:27:05-pubmed_qa-train-eval.out", "Cameron-BERT-eec-emotion-09:27:25-billsum-train-eval.out", 
	"Cameron-BERT-eec-emotion-09:28:44-tweet_eval-train-eval.out", "Cameron-BERT-eec-emotion-09:34:06-wikipedia-train-eval.out",
	"Cameron-BERT-Jigsaw-00:40:33-conv_ai_3-train-eval.out", "Cameron-BERT-Jigsaw-00:41:54-air_dialogue-train-eval.out", 
	"Cameron-BERT-Jigsaw-00:53:23-empathetic_dialogues-train-eval.out", "Cameron-BERT-Jigsaw-01:15:29-ted_talks_iwslt-train-eval.out", 
	"Cameron-BERT-Jigsaw-05:38:33-yahoo_answers_topics-train-eval.out", "Cameron-BERT-Jigsaw-09:26:33-pubmed_qa-train-eval.out", 
	"Cameron-BERT-Jigsaw-09:27:33-billsum-train-eval.out", "Cameron-BERT-Jigsaw-09:33:06-tweet_eval-train-eval.out", 
	"Cameron-BERT-Jigsaw-09:34:05-wikipedia-train-eval.out", "Cameron-BERT-mdgender-convai-binary-00:40:37-conv_ai_3-train-eval.out", 
	"Cameron-BERT-mdgender-convai-binary-00:42:04-air_dialogue-train-eval.out", "Cameron-BERT-mdgender-convai-binary-00:53:18-empathetic_dialogues-train-eval.out", 
	"Cameron-BERT-mdgender-convai-binary-01:30:30-ted_talks_iwslt-train-eval.out", "Cameron-BERT-mdgender-convai-binary-07:17:12-pubmed_qa-train-eval.out", 
	"Cameron-BERT-mdgender-convai-binary-09:27:16-billsum-train-eval.out", "Cameron-BERT-mdgender-convai-binary-09:30:01-wikipedia-train-eval.out", 
	"Cameron-BERT-mdgender-convai-binary-09:50:56-tweet_eval-train-eval.out", "Cameron-BERT-mdgender-convai-ternary-00:40:41-conv_ai_3-train-eval.out", 
 	"Cameron-BERT-mdgender-convai-ternary-00:42:00-air_dialogue-train-eval.out", "Cameron-BERT-mdgender-convai-ternary-00:44:00-tweet_eval-train-eval.out", 
 	"Cameron-BERT-mdgender-convai-ternary-00:53:56-empathetic_dialogues-train-eval.out", "Cameron-BERT-mdgender-convai-ternary-01:16:10-ted_talks_iwslt-train-eval.out", 
 	"Cameron-BERT-mdgender-convai-ternary-02:30:18-yahoo_answers_topics-train-eval.out", "Cameron-BERT-mdgender-convai-ternary-06:15:39-pubmed_qa-train-eval.out", 
 	"Cameron-BERT-mdgender-convai-ternary-09:27:16-billsum-train-eval.out", "Cameron-BERT-mdgender-convai-ternary-09:29:48-wikipedia-train-eval.out", 
 	"Cameron-BERT-mdgender-wizard-00:40:34-conv_ai_3-train-eval.out", "Cameron-BERT-mdgender-wizard-00:41:58-air_dialogue-train-eval.out", 
	"Cameron-BERT-mdgender-wizard-00:42:15-ted_talks_iwslt-train-eval.out", "Cameron-BERT-mdgender-wizard-00:43:40-tweet_eval-train-eval.out", 
	"Cameron-BERT-mdgender-wizard-00:53:08-empathetic_dialogues-train-eval.out", "Cameron-BERT-mdgender-wizard-01:14:09-ted_talks_iwslt-train-eval.out", 
	"Cameron-BERT-mdgender-wizard-02:17:49-yahoo_answers_topics-train-eval.out", "Cameron-BERT-mdgender-wizard-06:14:44-pubmed_qa-train-eval.out", 
	"Cameron-BERT-mdgender-wizard-09:24:57-billsum-train-eval.out", "Cameron-BERT-mdgender-wizard-09:34:07-wikipedia-train-eval.out", 
	"Cameron-BERT-mdgender-wizard-09:48:57-tweet_eval-train-eval.out", "Cameron-BERT-rtgender-opgender-annotations-00:40:32-conv_ai_3-train-eval.out",
	"Cameron-BERT-rtgender-opgender-annotations-00:42:00-air_dialogue-train-eval.out", "Cameron-BERT-rtgender-opgender-annotations-00:53:24-empathetic_dialogues-train-eval.out", 
	"Cameron-BERT-rtgender-opgender-annotations-01:17:11-ted_talks_iwslt-train-eval.out", "Cameron-BERT-rtgender-opgender-annotations-02:16:52-yahoo_answers_topics-train-eval.out",
	"Cameron-BERT-rtgender-opgender-annotations-05:26:25-pubmed_qa-train-eval.out", "Cameron-BERT-rtgender-opgender-annotations-09:24:49-billsum-train-eval.out", 
	"Cameron-BERT-rtgender-opgender-annotations-09:33:56-wikipedia-train-eval.out", "Cameron-BERT-rtgender-opgender-annotations-12:24:10-tweet_eval-train-eval.out", 
	"Cameron-BERT-SBIC-offensive-00:40:33-conv_ai_3-train-eval.out", "Cameron-BERT-SBIC-offensive-00:41:58-air_dialogue-train-eval.out", 
	"Cameron-BERT-SBIC-offensive-00:53:12-empathetic_dialogues-train-eval.out", "Cameron-BERT-SBIC-offensive-01:31:48-ted_talks_iwslt-train-eval.out", 
	"Cameron-BERT-SBIC-offensive-05:39:26-yahoo_answers_topics-train-eval.out", "Cameron-BERT-SBIC-offensive-09:26:27-pubmed_qa-train-eval.out", 
	"Cameron-BERT-SBIC-offensive-09:27:16-billsum-train-eval.out", "Cameron-BERT-SBIC-offensive-09:34:15-wikipedia-train-eval.out", 
	"Cameron-BERT-SBIC-offensive-09:49:24-tweet_eval-train-eval.out", "Cameron-BERT-SBIC-targetcategory-00:40:41-conv_ai_3-train-eval.out", 
	"Cameron-BERT-SBIC-targetcategory-00:42:00-air_dialogue-train-eval.out", "Cameron-BERT-SBIC-targetcategory-00:53:59-empathetic_dialogues-train-eval.out", 
	"Cameron-BERT-SBIC-targetcategory-01:17:08-ted_talks_iwslt-train-eval.out", "Cameron-BERT-SBIC-targetcategory-05:37:17-yahoo_answers_topics-train-eval.out", 
	"Cameron-BERT-SBIC-targetcategory-09:26:23-pubmed_qa-train-eval.out", "Cameron-BERT-SBIC-targetcategory-09:27:14-billsum-train-eval.out", 
	"Cameron-BERT-SBIC-targetcategory-09:35:03-wikipedia-train-eval.out", "Cameron-BERT-SBIC-targetcategory-10:03:40-tweet_eval-train-eval.out"]


def save_top_sentences(file_name):
	df = read_outfile(f"results/{file_name}", delimiter="|", skiprows=2)
	df = df.drop(columns=['predictions'])
	with open(f"sentences/{file_name}_preds", "a") as f:
		for i in tqdm(df.columns[1:]):
			df = df.sort_values(by=[i])
			f.write("\n")
			f.write(df.head(10).to_string())
			f.write(df.tail(10).to_string())
			f.write("\n")
	f.close()



for file_name in list_results:
	save_top_sentences(file_name)


from analysis import *


def save_description(file_name):
	with open(filename, "w") as output_file:
		df1 = read_outfile(args.input1)
		rfunc_1 = analysis_relabel_functions[args.relabel1]

		df1_temp = df1["predictions"].map(rfunc_1)
		df1_temp = pd.DataFrame(df1_temp.to_list(), columns=["scores_1", "category_1"])
		df1_temp["scores_1"] = df1_temp["scores_1"].map(lambda x: x.item())
		df1 = df1.join(df1_temp)

		df1_describe = df1["scores_1"].describe()

		output_file.write(f"Description of scores (unscaled) for data at {args.input1}\n{df1_describe}\n\n")	