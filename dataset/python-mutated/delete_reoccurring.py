"""
QUESTION: Given a string as your input, delete any reoccurring 
character, and return the new string.

This is a Google warmup interview question that was asked duirng phone screening
at my university. 
"""

def delete_reoccurring_characters(string):
    if False:
        while True:
            i = 10
    seen_characters = set()
    output_string = ''
    for char in string:
        if char not in seen_characters:
            seen_characters.add(char)
            output_string += char
    return output_string