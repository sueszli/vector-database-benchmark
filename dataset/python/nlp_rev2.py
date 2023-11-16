
def nlp_function(words):
    from keytotext import pipeline
    nlp = pipeline("k2t-new")
    sentence = nlp(words)
    message = f"{sentence} | Key words: {str(words)}"
    return message

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    translated_sentences = []

    for line in lines:
        words = line.strip().split()
        processed_result = nlp_function(words)
        translated_sentences.append(processed_result)

    translated_str = "\n".join(translated_sentences)

    with open(output_file, 'w') as f:
        f.write(translated_str)

    return translated_str


def main(input_file_path, output_file_path):
    translated_sentence = process_file(input_file_path, output_file_path)
    print(translated_sentence)
    return translated_sentence


if __name__ == "__main__":
    input_file_path = '/home/tester/finalProject/translated_content.txt'
    output_file_path = '/home/tester/finalProject/translated_content.txt'
    main(input_file_path, output_file_path)
