import nltk

nltk.download("punkt")
import re
import pandas as pd

textbook_pdf_paths = ["./textbooks/os.txt",
    "./textbooks/cn.txt",
    "./textbooks/ooad.txt",
    "./textbooks/se1.txt",
    "./textbooks/se2.txt"]

#df = pd.DataFrame(columns=["Question", "Answer", "Context"])
df_index = 0
outf = open("questions.txt", "w")

for name in textbook_pdf_paths:
    f = open(name, "r")

    textbook = f.read()

    textbook = re.sub("\n\n[^\n]*\n\n", "\n", textbook)
    textbook = re.sub("\n([0-9]+.)*[0-9]\n", "\n", textbook)
    textbook = re.sub("\nFigure.*\n", "\n", textbook)
    textbook = textbook.encode("ascii", errors="ignore").decode()
    textbook = re.sub("\n[ ]", "\n", textbook)
    textbook = re.sub("\n.[0-9]+\n", "\n", textbook)
    textbook = re.sub("\nChapter.*\n", "\n", textbook)
    textbook = re.sub("^\\s*U\\+\\w+\\s*", "", textbook)

    words = ["For example", "For instance", "For eg.", "To give an example", "Such as"]
    question_count = 0

    # FINDING THE ANSWER

    for word in words:
        for match in re.finditer(word, textbook):
            start = match.span(0)[0]  # index of first letter of words
            end = start = match.span(0)[1]  # index of last letter of words

            answer = ""
            question = "Justify with an example how"

            # Adding the sentence before the key-phrase

            temp = ""
            flag = 0
            x = end - len(word) - 1  # index of first character in 'words'
            while x >= 0:
                if textbook[x] == "." and flag:
                    break
                elif (
                    textbook[x] == "." and not flag
                ):  # First full stop encountered ignored
                    flag = 1
                else:
                    temp = (
                        textbook[x] + temp
                    )  # keep adding words from the end of the sentence to the start of the sentence
                x -= 1  # Moving in the following order: The <-- reason <-- this <-- .....
            answer += temp
            answer = answer.replace("\n", " ")
            answer = re.sub(" +", " ", answer)

            # Adding the sentence after the key-phrase
            answer += "." + word
            x = end
            while x < len(textbook):
                if textbook[x] == "." and textbook[x + 2].isupper():
                    break
                else:
                    answer += textbook[x]
                x += 1
            answer = answer.replace("\n", " ")

            # FINDING THE QUESTION - Sentence preceding the 'For example'

            temp = ""
            flag = 0
            x = end - len(word) - 1  # index of first character in 'words'
            while x >= 0:
                if textbook[x] == "." and flag:
                    # if(textbook[x].isupper() and textbook[x-1]=="\n"): #if x is the starting of a sentence
                    # temp = textbook[x] + temp
                    temp = temp[0].lower() + temp[1].lower() + temp[2:]
                    break
                elif (
                    textbook[x] == "." and not flag
                ):  # First full stop encountered ignored
                    flag = 1
                else:
                    temp = (
                        textbook[x] + temp
                    )  # keep adding words from the end of the sentence to the start of the sentence
                x -= 1  # Moving in the following order: The <-- reason <-- this <-- .....
            question += temp

            # FINDING THE CONTEXT

            context = ""
            x = end  # From the 'word' to the end of the paragraph
            para_count = 0
            while x < len(textbook):
                if textbook[x] == "." and textbook[x + 1] == "\n":
                    para_count += 1
                    if para_count == 2:
                        break
                context += textbook[x]
                x += 1

            x = end - 1  # From the 'word' to the start of the paragraph
            para_count = 0
            while x >= 0:
                if textbook[x] == "." and textbook[x + 1] == "\n":
                    para_count += 1
                    if para_count == 2:
                        break
                context = textbook[x] + context
                x -= 1

            answer = answer.replace("\n", " ")
            answer = re.sub(" +", " ", answer)

            question = question.replace("\n", " ")
            context = context.replace("\n", " ")
            context = re.sub(" +", " ", context)

            if len(answer.strip()) == 0:
                continue

            question_count += 1
            outf.write("\n\nQuestion : " + question)
            outf.write("\n\nAnswer : " + answer)
            outf.write("\n\nContext : " + context)
            #df.loc[df_index] = [question, answer, context]
            df_index += 1

    print("Successfully generated ", df_index, " questions")

#df.to_csv("justify_with_eg_without_cn.csv")
