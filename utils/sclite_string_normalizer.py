from num2words import num2words
import re
import unidecode

personal_names_dict = {
    # incorrect version : # correct version
    'cas' : 'kas',
    'sophie' : 'sofie',
    'mathias' : 'matthias',
    'lukas' : 'lucas', 
    'tes' : 'tess',
    'bill' : 'bil',
    'lisbeth' : 'liesbeth',
    'jez' : 'jess',
    'zara' : 'sarah',
    'robby' : 'robbie',
    'vledder' : 'fledder',
    'mathijs' : 'matthijs',
    'rosemarijn' : 'rozemarijn',
}

speaker_sounds_dict = {
    # incorrect version : # correct version
    'sssst' : 'sst',
    'ssst' : 'sst',
    'psst' : 'pst',
    'pssst' : 'pst',
    'wow' : 'wauw'
}

spelling_variants_with_dashes_dict = {
    'yo-yo': 'jojo',
    'yo-yos': 'jojo\'s',
    'yo-yo\'s': 'jojo\'s',
}

spelling_variants_dict = {
    'yoyo': 'jojo',
    'yoyos': 'jojo\'s',
    'yoyo\'s': 'jojo\'s',
    'hardstikke': 'hartstikke',
    'giechelbril' : 'giegelbril',
    'snout' : 'snauwt',
    'bisons' : 'bizons',
    'bison' : 'bizon',
    'gevokt' : 'gefokt',
    'hooft' : 'hoofd',
    'schroefendraaier' : 'schroevendraaier',
    'beeltjes' : 'beeldjes'
}


def write_names_as_prompt_and_correct_spelling_variants(word):
    if word in list(personal_names_dict.keys()):
        return personal_names_dict[word]
    
    if word in list(spelling_variants_dict.keys()):
        return spelling_variants_dict[word] 

    if word in list(speaker_sounds_dict.keys()):
        return speaker_sounds_dict[word] 
    
    return word

def normalize_spellings_with_dashes(word):
    if word in list(spelling_variants_with_dashes_dict.keys()):
        return spelling_variants_with_dashes_dict[word]
    else:
        return word.replace('-', ' ')


"""
Replace " ‘ ’ ` with '
Only keep . ! ? ' - 
"""
def normalizeApostrophe(s):
    return "".join(['\'' if letter in '\'"‘’`' else letter for letter in s])

def onlyKeepOrthTransPunct(s):
    s = normalizeApostrophe(s)
    return "".join(['' if letter in '''!-'.?''' else letter for letter in s])

def removeAllPunctuation(s):
    punc = '''!()[]{};:'"\,<>./???@#$%^&*_‘~’'''
    # s = s.replace('-', ' ')
    return "".join([letter for letter in s if letter not in punc])

def removeAnnotations(ot):

    # Remove *a, *x, etc.
    return re.sub('(\*[a-z]){1}', '', ot)

def normalizePossessivePronouns(word):
    if word in ['zn', 'z\'n']:
        return 'zijn'
    elif word in ['mn', 'm\'n']:
        return 'mijn'
    return word

def correctApostropheSpellingErrorAsrTranscript(word):
    # If word starts with s'
    if word[0:2] == 's\'':
        return '\'s ' + word[2:]
    return word

# In three TextGrids of AVI-9 story 1 VL, België was written as Belgiî. I corrected this in the original TextGrids and in orthographicTranscriptionsDF


"""
Function to normalize a string
"""
def normalize_string(sentence: str = '', all_punct: bool = True, basic_punct: bool=False, lower: bool = True, accents: bool = True, number_to_letter : bool = True, names_as_prompt: bool = True, poss_pro: bool = True, annTags: bool = False, apostrophe_spelling_error:bool = True):
    if annTags:
        sentence = removeAnnotations(sentence)

    if lower:
        sentence = sentence.lower()

    normalized_word_list = []
    for word in sentence.split(' '):

        word = normalize_spellings_with_dashes(word)

        if apostrophe_spelling_error:
            word = correctApostropheSpellingErrorAsrTranscript(word)

        if all_punct:
            word = removeAllPunctuation(word)
            
        if basic_punct:
            word = onlyKeepOrthTransPunct(word)

        if number_to_letter:
            try:
                word = num2words(word.strip(), lang='nl')
            except:
                word = word

        if names_as_prompt:
            word = write_names_as_prompt_and_correct_spelling_variants(word)

        if poss_pro:
            word = normalizePossessivePronouns(word)

        normalized_word_list.append(word)

    new_sentence = " ".join([x.strip() for x in normalized_word_list])

    if accents:
        new_sentence = unidecode.unidecode(new_sentence)
    
    return new_sentence.strip()

# normalized_string = normalize_string('Hallo! Ik*u ben van-vandaag hier*a met 50 en 1 persoon, waaronder m\'n Cas en Lucas. Hoe is \'t?', True, False, True, True, True, True, True, True)
# print('\n', normalized_string)


