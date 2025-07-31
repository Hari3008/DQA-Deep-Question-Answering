import spacy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import sys
import codecs
import json
import re
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
from nltk import word_tokenize
import Levenshtein
from collections import Counter
import nltk

nltk.download("averaged_perceptron_tagger")
from nltk.tokenize import word_tokenize, sent_tokenize
import wikipedia
from nltk.stem.porter import *

stemmer = PorterStemmer()
spacy_model = spacy.load("en")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
sr_labeler = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)
coref = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
)


# functions
def print_graph(n, e):
    G = nx.Graph()
    G.add_nodes_from(n)
    G.add_edges_from(e)
    pos = nx.spring_layout(G, k=0.4)
    plt.figure(3, figsize=(30, 30))
    nx.draw(
        G,
        pos,
        width=2,
        linewidths=1,
        node_size=3000,
        node_color="pink",
        alpha=0.9,
        labels={
            node[0]: node[1]["text"] + ":" + str(node[1]["index"])
            for node in G.nodes(data=True)
        },
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={
            (edge[0], edge[1]): edge[2]["role"] for edge in G.edges(data=True)
        },
        font_color="red",
    )
    plt.show()
    return G


def connect_trees(nodes, edges):
    for node1 in nodes:
        for node2 in nodes:
            if node1[0] < node2[0]:
                sentence1 = str(node1[1]["text"])
                sentence2 = str(node2[1]["text"])
                ## encode sentences to get their embeddings
                # embedding1 = similarity_model.encode(sentence1, convert_to_tensor=True)
                # embedding2 = similarity_model.encode(sentence2, convert_to_tensor=True)
                # compute similarity scores of two embeddings
                # cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
                # print("Similarity score:", cosine_scores.item())
                # if(cosine_scores.item()>=0.75):
                # print(node1[1]['tok'],"-",node2[1]['tok'])
                if sentence1 == sentence2:
                    edges.append((node1[0], node2[0], {"role": "similar"}))

    return nodes, edges


def compute(xx):
    xx = xx.encode("utf-8", errors="ignore").decode("utf-8")
    xx = xx.replace("\n", " ")
    xx = xx.replace("'", "")
    xx = xx.replace('"', "")
    # xx=xx.replace("[","")
    # xx=xx.replace("]","")
    xx = xx.replace("/", "")
    xx = xx.replace("\\", "")
    xx = re.sub(" +", " ", xx)
    xx = re.sub("\[[0-9]\]", "", xx)
    # print(xx)

    sentence_span_dic = {}

    parse = spacy_model(xx)
    sentences = list(parse.sents)
    sent_span_count = 0
    for sentence in sentences:
        sentence_span_dic[sentence.text] = (
            sent_span_count,
            sent_span_count + len(sentence) - 1,
        )
        sent_span_count += len(sentence) + 1

    # print(sentence_span_dic)

    try:
        sent = sr_labeler.predict(sentence=xx)
    except Exception as e:
        print(e)
    length, words, verbs = len(sent["words"]), sent["words"], sent["verbs"]
    tags = [verb["tags"] for verb in verbs]

    # print(tags)

    try:
        sent2 = coref.predict(xx)
    except Exception as e:
        print(e)

    coref_clusters = sent2["clusters"]
    coref_links = {}

    for cluster in coref_clusters:
        main_node = tuple(cluster[0])
        for node in cluster[1:]:
            coref_links[tuple(node)] = main_node

    # print(coref_links)

    span_dics = []
    tokenized_document = sent2["document"]

    for role in sent["verbs"]:
        span_dic = {}
        tok_doc = tokenized_document
        des = role["description"]
        # print(des)
        count = 0
        index = 0
        while index < len(tok_doc):
            tok = tok_doc[index]
            if des.startswith(tok):
                des = des[len(tok) + 1 :]
                count += 1
                index += 1
            else:
                # print(des)
                # try:
                # print(des)
                if not des.startswith("[ARG") and not des.startswith("[V"):
                    des = des[1:]
                    continue
                node_type = des[: des.index(":")][1:]
                if not node_type.startswith("ARG") and node_type != "V":
                    node_type = node_type[node_type.index("-") + 1 :]
                des = des[des.index(":") + 2 :]
                # except Exception as e:
                #  des=des[1:]
                start = count
                # print(des)
                # print(tok_doc[index])
                while index < len(tok_doc) and des.startswith(tok_doc[index]):
                    des = des[len(tok_doc[index]) + 1 :]
                    # print(tok_doc[index])
                    # print(des)
                    index += 1
                    count += 1
                des = des[1:]
                end = count - 1
                span_dic[node_type] = (start, end)

        span_dics.append(span_dic)

    # print(span_dics)

    graph_data = []

    for tag in tags:
        dic = {}
        for i in range(len(tag)):
            if "-" not in tag[i]:
                continue
            node_type = tag[i][tag[i].index("-") + 1 :]
            if not node_type.startswith("ARG") and node_type != "V":
                node_type = node_type[node_type.index("-") + 1 :]
            dic[node_type] = dic.get(node_type, "") + words[i] + " "
        # if("O" in dic):
        #  del dic["O"]
        for key in dic:
            dic[key] = dic[key].strip()
        graph_data.append(dic)

    # print(graph_data)

    nodes = []
    edges = []

    count = 0
    track = {}
    track2 = {}
    for index, x in enumerate(graph_data):
        if len(x.keys()) == 1:
            continue
        nodes.append(
            (
                count,
                {
                    "text": x["V"],
                    "index": index,
                    "verb": "yes",
                    "span_start": span_dics[index]["V"][0],
                    "span_end": span_dics[index]["V"][1],
                },
            )
        )
        verb_index = count
        count += 1

        for key in x.keys():

            if key == "V":
                continue

            min_edit_distance_track2_key = None
            for a in track2:
                sentence1 = str(x[key])
                sentence2 = str(a)
                embedding1 = similarity_model.encode(sentence1, convert_to_tensor=True)
                embedding2 = similarity_model.encode(sentence2, convert_to_tensor=True)

                cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

                if cosine_scores.item() >= 0.94:
                    min_edit_distance_track2_key = a
                    break
                # if(Levenshtein.distance(x[key].lower(),a.lower())/max(len(x[key]),len(a))<0.25):
                #  print(Levenshtein.distance(x[key].lower(),a.lower())/max(len(x[key]),len(a)))
                #  min_edit_distance_track2_key=a
                #  break

            min_edit_distance_track_key = None

            if span_dics[index][key] in coref_links:
                for a in track:
                    spans = sorted([coref_links[span_dics[index][key]], a])
                    if spans[0][1] <= spans[1][0]:
                        continue
                    else:
                        overlap = spans[0][1] - spans[1][0]
                        if (
                            overlap
                            / max(spans[0][1] - spans[0][0], spans[1][1] - spans[1][0])
                            > 0.85
                        ):
                            min_edit_distance_track_key = coref_links[
                                span_dics[index][key]
                            ]
                            break

            if (
                span_dics[index][key] in coref_links
                and min_edit_distance_track_key != None
            ):  # coref_links[span_dics[index][key]] in track):
                # print(track)
                # print(x[key],"Combined using Coreference with ",min_edit_distance_track_key,track[min_edit_distance_track_key])
                edges.append(
                    (track[min_edit_distance_track_key], verb_index, {"role": key})
                )
            elif min_edit_distance_track2_key != None:
                # print(x[key],"combined with ",min_edit_distance_track2_key.lower())
                edges.append(
                    (
                        track2[min_edit_distance_track2_key.lower()],
                        verb_index,
                        {"role": key},
                    )
                )
            else:
                nodes.append((count, {"text": x[key], "index": "", "verb": "no"}))
                track[span_dics[index][key]] = count
                edges.append((count, verb_index, {"role": key}))
                track2[x[key].lower()] = count
                count += 1
            """
      if(span_dics[index][key] in coref_links and  coref_links[span_dics[index][key]] in track):
        edges.append((track[coref_links[span_dics[index][key]]],verb_index,{'role':key}))
      elif(x[key].lower() in track2):
        edges.append((track2[x[key].lower()],verb_index,{"role":key}))
      else:
        nodes.append((count,{"text":x[key],"index":"","verb":"no","span_start":span_dics[index][key][0],"span_end":span_dics[index][key][1]}))
        track[span_dics[index][key]]=count
        edges.append((count,verb_index,{"role":key}))
        track2[x[key].lower()]=count
        count+=1
      """

    # nodes,edges=connect_trees(nodes,edges)

    # print(nodes)
    G = print_graph(nodes, edges)

    track_dic = {}
    track_dic2 = {}

    stative_verbs = [
        "is",
        "are",
        "were",
        "have",
        "has",
        "had",
        "be",
        "belong",
        "own",
        "contain",
        "include",
        "consist",
        "possess",
        "lack",
    ]

    def generate_question(text, args, verbs):
        counts = Counter(args)
        stative_count = 0
        not_past_tense_verbs = {}

        for index, verb in enumerate(verbs):
            tokenized = sent_tokenize(verb)
            for i in tokenized:
                wordsList = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(wordsList)
                if (
                    verb not in stative_verbs
                    and "VBG" not in tagged
                    and "VBN" not in tagged
                    and "VBG" not in tagged
                ):
                    not_past_tense_verbs[args[index]] = (
                        not_past_tense_verbs.get(args[index], 0) + 1
                    )
                # print(tagged)

        for verb in verbs:
            if verb in stative_verbs:
                stative_count += 1

        use_count = 0

        for verb in verbs:
            if verb in ["use", "used", "uses", "using"]:
                use_count += 1

        if use_count > 2:
            return "List the uses of {}".format(text)

        if 0 < stative_count <= 2 and stative_count >= 0.5 * len(verbs):
            return "What is " + text.lower()

        if stative_count >= 0.5 * len(verbs):
            tokenized = sent_tokenize(text)
            for i in tokenized:
                wordsList = nltk.word_tokenize(i)
                count = 0
                stem = stemmer.stem(wordsList[-1])
                if len(wordsList[-1]) - len(stem) >= 2:
                    return "Explain " + text.lower()
            return "Describe " + text.lower()

        if (
            counts["ARG0"] > 0.7 * len(verbs)
            and not_past_tense_verbs["ARG0"] > 0.75 * counts["ARG0"]
        ):
            return "Describe the role of " + text.lower()

        if counts["ARG1"] > 3 and counts["ARG1"] > 0.75 * len(verbs):
            return "List what happens to " + text.lower()

        # if(counts["ARG0"]>0.4*len(verbs)):
        #  return("Explain "+text)

        # return("List what happens to "+text)

    questions = []
    answers = []
    qanda = []
    track_dic3 = {}
    track_dic4 = {}

    for edge in G.edges(data=True):
        if G.nodes[edge[0]]["verb"] == "no":
            track_dic[edge[0]] = track_dic.get(edge[0], []) + [edge[2]["role"]]
            track_dic2[edge[0]] = track_dic2.get(edge[0], []) + [
                G.nodes[edge[1]]["text"]
            ]
            track_dic3[edge[0]] = track_dic3.get(edge[0], []) + [G.nodes[edge[1]]]
        else:
            track_dic[edge[1]] = track_dic.get(edge[1], []) + [edge[2]["role"]]
            track_dic2[edge[1]] = track_dic2.get(edge[1], []) + [
                G.nodes[edge[0]]["text"]
            ]
            track_dic3[edge[1]] = track_dic3.get(edge[1], []) + [G.nodes[edge[0]]]

    for key in track_dic:
        count = 0
        for val in track_dic[key]:
            if re.search("^ARG[0-4]$", val):
                count += 1
        if count >= 2:
            track_dic[key] = [
                ba[1]
                for ba in sorted(
                    zip([ab["index"] for ab in track_dic3[key]], track_dic[key])
                )
            ]
            track_dic2[key] = [
                ba[1]
                for ba in sorted(
                    zip([ab["index"] for ab in track_dic3[key]], track_dic2[key])
                )
            ]
            track_dic3[key] = [
                ba[1]
                for ba in sorted(
                    zip([ab["index"] for ab in track_dic3[key]], track_dic3[key])
                )
            ]
            print(
                G.nodes[key]["text"], track_dic[key], track_dic2[key], track_dic3[key]
            )
            y = generate_question(G.nodes[key]["text"], track_dic[key], track_dic2[key])
            if y != None:
                answer = {}
                sentence_count = 0
                for c in track_dic3[key]:
                    for sentence in sentence_span_dic:
                        # G.nodes[key]["span_start"] >= sentence_span_dic[sentence][0] and G.nodes[key]["span_end"] <= sentence_span_dic[sentence][1]
                        if (
                            c["span_start"] >= sentence_span_dic[sentence][0]
                            and c["span_end"] <= sentence_span_dic[sentence][1]
                        ):
                            if sentence not in answer:
                                answer[sentence] = sentence_count
                                sentence_count += 1
                            # print(sentence_span_dic[sentence],G.nodes[key]["span_start"],G.nodes[key]["span_end"],c["span_start"],c["span_end"],"Matched")
                questions.append(y)
                complete_answer = "".join(
                    [aa[0] for aa in sorted(answer.items(), key=lambda temp: temp[1])]
                )
                answers.append(complete_answer)
                qanda.append({"question": y, "answer": complete_answer})
    return qanda


def generate_dataset(list_of_titles):
    # take list of potential titles
    # for each title generate 5 suggestions
    global df
    counter = 0
    for title in list_of_titles:
        # for every suggestions, run func. to generate questions
        suggestions = wikipedia.search(title)
        for suggestion in suggestions:
            try:
                suggestion_summary = wikipedia.summary(suggestion)
                # print(suggestion_summary)
                qandadic = compute(suggestion_summary)
                # print(qandadic)
                for dic in qandadic:
                    dic["context"] = suggestion_summary
                print(qandadic)
                if len(qandadic) > 0:
                    df = df.append(qandadic, ignore_index=True)
                    if counter % 10 == 0:
                        df.to_csv("/content/drive/MyDrive/dataset24.csv")
            except Exception as e:
                continue
            counter += 1
        print("\nTitle : ", title, " ------------------ Done :)\n\n")
        print()


file = open("./generator_1_input.txt", mode="r")
context = file.read()
file.close()

print()
print(
    "---------------------------------------------------------------------------------"
)
for qac in compute(context):
    print()
    print()
    print("Question : ", qac["question"])
    print()
    print("Answer : ", qac["answer"])
    print()
    print()
    print(
        "---------------------------------------------------------------------------------"
    
