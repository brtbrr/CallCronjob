import re
from bs4 import BeautifulSoup



def find_words(soup):
    """
    return:
        words - list of dicts
    """
    channels = soup.find_all('channel')
    words = []
    for i, channel in enumerate(channels):
        cur_words = channel.find_all('word')
        words += [{'channel': i, 'text': word.get('text'), 'start_time': int(word.get('startms'))}
                  for word in cur_words]

    words.sort(key=lambda x: x['start_time'])
    return words


def get_replics(words):
    all_phrases = []

    cur_phrase = words[0]['text']
    cur_channel = start_channel = words[0]['channel']
    for word in words[1:]:
        if word['channel'] == cur_channel:
            cur_phrase += ' ' + word['text']
        else:
            all_phrases.append(cur_phrase)
            cur_phrase = word['text']
            cur_channel = word['channel']
    all_phrases.append(cur_phrase)
    return all_phrases, start_channel


def make_dialog(text):
    """
    text - xml
    return:
        all_replics - все реплики
        start_channel - кто начал разговор. 1 - СОТРУДНИК. 0 - КЛИЕНТ
    """

    soup = BeautifulSoup(text)
    words = find_words(soup)
    if not words:
        return None, None
    all_phrases, start_channel = get_replics(words)

    return all_phrases, start_channel


def crt_preprocessing(call):
    lst_text, is_outgoing = make_dialog(call)

    if lst_text is None:
        return None

    text = " ".join(lst_text[1:])

    words = ["*****", "*****", ......, "*****", "*******"]

    for word in words:
        text = re.sub(fr'{word}', ' ', text)

    text = re.sub(r'[0-9]+', '', text)

    return text


def replace_all(text, dic):
    for i, j in dic.items():
        text = re.sub(i.lower(), j, text.lower())

    return text


def note_preprocessing(text):
    text = re.sub(r"https?://[^,\s]+,?", "", str(text))

    text = ' '.join(re.findall(r'\w+', text))

    replacements = {fr'\b{re.escape(key)}\b': value for key, value in
                    abbreviations.items()}

    text = replace_all(text, replacements)

    text = re.sub(r'[0-9]+', '', text)

    return text.lower()

abbreviations = {
    "***": "*********",
    "***": "*********",
    "***": "*********"
}

