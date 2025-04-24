"""
This is a new algorithm, created to replace the ADAGT algorithm.
This algorithms is faster and better in keeping words together in case there are many deletions (the child forgot to read the words)

New alignment:
ro-de doppen daarna laat ze op het bord een tekening zien daarop staat
ronde do---n----------------------------------ken--- -----daarop staat

Old ADAPT alignment:
rode doppen daarna laat ze op het bord een tekening zien daarop staat
ro--------n-d------------e ----------d----------onk --en-daarop staat

"""

from Bio import Align
from string2string.alignment import NeedlemanWunsch
import os
import re
import pandas as pd

nw = NeedlemanWunsch()

def split_alignments_in_segments(align_ref, align_hyp):

    # indices_rev = [0] + [i.start() for i in re.finditer(" ",align_ref_rev)] + [len(align_ref_rev)] + [9999]
    # align_ref_rev_list = [align_ref_rev[indices_rev[idx]: indices_rev[idx+1]]
    #                       for idx, item in enumerate(indices_rev) if item != 9999][:-1]
    # align_hyp_rev_list = [align_hyp_rev[indices_rev[idx]: indices_rev[idx+1]]
    #                       for idx, item in enumerate(indices_rev) if item != 9999][:-1]

    indices = [0] + [i.start() for i in re.finditer(" ", align_ref)
                     ] + [len(align_ref)] + [9999]
    align_ref_list = [align_ref[indices[idx]: indices[idx+1]]
                      for idx, item in enumerate(indices) if item != 9999][:-1]
    align_hyp_list = [align_hyp[indices[idx]: indices[idx+1]]
                      for idx, item in enumerate(indices) if item != 9999][:-1]


    align_ref_list = [align_ref_list[0]] + [x[1:] for x in align_ref_list[1:]]
    align_hyp_list = [align_hyp_list[0]] + [x[1:] for x in align_hyp_list[1:]]

    return align_ref_list, align_hyp_list

"""
If either aligned_asrTrans or reversed_aligned_asrTrans contains 'prompt', the word is said to be pronounced correctly.
"""
def determineCorrectness(row):
    return row['prompt'] in row['aligned_asrTrans']

# def getPromptIdxs(basename, pathToPromptIdxs):

#     task = basename.split('-')[1]
#     taskType = task.split('_')[0]
#     taskNr = task.split('_')[1]

#     promptFileName = task + '-wordIDX.csv'
#     promptFile = os.path.join(pathToPromptIdxs, promptFileName)

#     promptDF = pd.read_csv(promptFile)

#     return list(promptDF['prompt_id'])

def one_way_alignment_modern(prompt, asrTrans):
    ############
    # Step 1: Global alignment using Bio aligner. 
    # We chose this one, since it keeps asrTrans words as much as possible together, example alignment:
    # ------jo-j-o met twee doppen--- van--- een fles- kun-- je bijvoorbeeld een jojo-- maken -------wat is een jojo-- nou w-eer vraagt t-eun dat is heel erg leuk -ouderwets spee--lgo-ed legt juf- s-oets- uit ze laat een jojo---- zien en doet voor hoe het werkt die jojo---- heeft ze zelf gemaakt van twee ro-de doppen daarna laat ze op het bord een tekening zien daarop staat precies wat ze moeten doen om een jojo---- te maken dan l-egt ze heel veel gekleurde dopp--en op haar --tafel nu mogen jull--ie het allemaal zelf proberen ze-----gt ze de-- klas- is- er- de hele middag d---ruk mee- b-e-zig aan het eind- van de dag hebben- ze --samen wel dertig jojos gemaakt ik ben echt ontzettend trots op jullie zegt de juf lachend daarom heb ik een l-euke verrassing voor jullie ze zwaait- naar iemand die op de gang staat de deur gaat open en meneer zemel komt binnen hallo kinderen zegt hij vrolijk ik hoor van de juf dat jullie zo goed helpen met speelgoed maken hij loopt door de klas en bekijkt alle jojos
    # begin jol-lo met twee do---nker v--oor een f-est k--om je bijvoorbeeld een jo--ng ma--n krijgt wat is een jo--ng nou -meer vraagt -zeun dat is heel erg leuk nou------- s---til--ze-----t j--e stoe--l uit ze laat een jo--ngen zien en doet voor hoe het werkt d-e jo--ngen heeft ze zelf gemaakt van twee ronde do---n----------------------------------ken--- -----daarop staat precies wat ze moet-- doen om een jo--ngen te maken dan -zegt ze heel veel gekleurde do--nken op haar zetafel nu mogen ju--ffie het allemaal zelf proberen zelf zegt ze d-at k-a-n i-k erg de hele middag de druk me-t -de zi----n het einde van de dag he----t z--ilsamen -------------------m---------e-----t ---------d---------------e --------j---------------o-----------n--ge------------n- ------------z------i-j -------ma--------------a-------------------------------------------k--t ----------------------z-----i-----l------------v----e----------------------------------------------------------------r------------------------------
    ###########

    # Initialize the aligner, create alignments, and select the first one.
    aligner = Align.PairwiseAligner(match_score=1.0)
    aligner.match_score = 1.0
    alignments = aligner.align(prompt, asrTrans)
    al = alignments[0]
    align_ref = al[0]
    align_hyp = al[1]

    ############
    # Step 2: Split aligned strings at clear word boundaries (a space at the same spot in both align_ref and align_hyp)
    ############
    index_list = []
    align_ref = ' ' + align_ref 
    align_hyp = ' ' + align_hyp

    for idx in range(len(align_ref)):

        ref_char_at_idx = align_ref[idx]
        hyp_char_at_idx = align_hyp[idx]

        if ref_char_at_idx == ' ' and hyp_char_at_idx == ' ':
            index_list.append(idx+1)

    align_ref_split = [align_ref[i:j].strip() for i,j in zip(index_list, index_list[1:]+[None])]
    align_hyp_split = [align_hyp[i:j].strip() for i,j in zip(index_list, index_list[1:]+[None])]

    # print('version1')
    # print(align_ref_split)
    # print(align_hyp_split)

    #############
    # STEP 3: Align these segments again, this time use word-level alignment.
    # In this way, hyp words are not spread out over multiple ref words, but stay more together. 
    # I hope that this will improve reconstruction of the confidence scores later on.
    #########

    alignment = [nw.get_alignment(i.replace('-', '').split(' '), j.replace('-', '').split(' ')) for i,j in zip(align_ref_split, align_hyp_split)]

    align_ref_2 = [x[0].split(' | ') for x in alignment]
    align_hyp_2 = [x[1].split(' | ') for x in alignment]

    align_ref_flatten = [item for sublist in align_ref_2 for item in sublist]
    align_hyp_flatten = [item for sublist in align_hyp_2 for item in sublist]

    # print('version2')
    # print(align_ref_flatten)
    # print(align_hyp_flatten)

    # print([(i.replace('-', '').strip(), j.replace('-', '').strip()) for i,j in zip(align_ref_flatten, align_hyp_flatten)])

    ##############
    # STEP 4: Local alignments
    # Now each word in the hypothesis is matched with a word in the reference, we can align these segments again, to get better alignment.
    # ['-----', 'jo-jo', 'met', 'twee', 'doppen', 'v-an', 'een', 'fles-', 'kun', 'je', 'bijvoorbeeld', 'een', 'jojo', 'maken', '------', 'wat', 'is', 'een', 'jojo', 'nou', 'weer', 'vraagt', 'teun', 'dat', 'is', 'heel', 'erg', 'leuk', '-ouderwets', 'speelgoed', '---l-egt', 'juf', 's-oets', 'uit', 'ze', 'laat', 'een', 'jo--jo', 'zien', 'en', 'doet', 'voor', 'hoe', 'het', 'werkt', 'die', 'jo--jo', 'heeft', 'ze', 'zelf', 'gemaakt', 'van', 'twee', 'ro-de', 'doppen', 'daarna', 'laat', 'ze', 'op', 'het', 'bord', 'een', '-tekening', 'zien', 'daarop', 'staat', 'precies', 'wat', 'ze', 'moeten', 'doen', 'om', 'een', 'jo--jo', 'te', 'maken', 'dan', 'legt', 'ze', 'heel', 'veel', 'gekleurde', 'doppen', 'op', 'haar', '--tafel', 'nu', 'mogen', 'jullie', 'het', 'allemaal', 'zelf', 'proberen', '----', 'zegt', 'ze', 'd-e', 'klas', 'is', 'er-', 'de', 'hele', 'middag', '--', 'druk', 'mee', 'bezig', 'aan', 'het', 'eind-', 'van', 'de', 'dag', 'hebben', 'ze', '---samen', 'wel', 'dertig', 'jojos', 'gemaakt', 'ik', 'ben', '-echt', 'ontzettend', 'trots', 'op', 'jullie', 'zegt', 'de', 'juf', 'lachend', 'daarom', 'heb', 'ik', 'een', 'leuke', 'verrassing--', 'voor', 'jullie', 'ze', 'zwaait', 'naar', 'iemand', 'die', 'op', 'de', 'gang', 'staat', 'de', 'deur', 'gaat', 'open', 'en', 'meneer', 'zemel', '-komt', 'binnen', 'hallo', 'kinderen', 'zegt', 'hij', 'vrolijk', 'ik', 'hoor', 'van', 'de', 'juf', 'dat', 'jullie', 'zo', 'goed', 'helpen', 'met', 'speelgoed', 'maken', 'hij', 'loopt', 'door', 'de', 'klas', 'en', 'bekijkt', 'alle', '-jojos']
    # ['begin', 'jollo', 'met', 'twee', 'donker', 'voor', 'een', 'f-est', 'kom', 'je', 'bijvoorbeeld', 'een', 'jong', 'ma--n', 'krijgt', 'wat', 'is', 'een', 'jong', 'nou', 'meer', 'vraagt', 'zeun', 'dat', 'is', 'heel', 'erg', 'leuk', 'nou-------', '---------', 'stilze-t', 'j-e', 'stoe-l', 'uit', 'ze', 'laat', 'een', 'jongen', 'zien', 'en', 'doet', 'voor', 'hoe', 'het', 'werkt', 'd-e', 'jongen', 'heeft', 'ze', 'zelf', 'gemaakt', 'van', 'twee', 'ronde', '------', '------', '----', '--', '--', '---', '----', '---', 'donke--n-', '----', 'daarop', 'staat', 'precies', 'wat', 'ze', 'moet--', 'doen', 'om', 'een', 'jongen', 'te', 'maken', 'dan', 'zegt', 'ze', 'heel', 'veel', 'gekleurde', 'donken', 'op', 'haar', 'zetafel', 'nu', 'mogen', 'juffie', 'het', 'allemaal', 'zelf', 'proberen', 'zelf', 'zegt', 'ze', 'dat', 'k-an', 'ik', 'erg', 'de', 'hele', 'middag', 'de', 'druk', 'met', 'de---', 'zin', 'het', 'einde', 'van', 'de', 'dag', 'h---et', '--', 'zilsamen', '---', '------', '-----', '-------', '--', '---', 'me--t', '----------', '-----', '--', '----de', '----', '--', '---', '-------', '------', '---', '--', '---', '-----', '------jongen', '----', '------', '--', 'z---ij', '----', '------', '---', '--', '--', '----', '-----', '--', '----', '----', '----', '--', '------', '-----', 'maakt', '------', '-----', '--------', '----', '---', '-------', '--', '----', '---', '--', '---', '---', '------', '--', '----', '------', '---', '---------', '-----', '---', '-----', '----', '--', '----', '--', '-------', '----', 'zilver']
    ##############

    alignment2 = [nw.get_alignment(i.replace('-', '').strip(), j.replace('-', '').strip()) for i,j in zip(align_ref_flatten, align_hyp_flatten)]

    align_ref_3 = [x[0].replace(' | ', '') for x in alignment2]
    align_hyp_3 = [x[1].replace(' | ', '') for x in alignment2]

    # print('version3')
    # print(align_ref_3)
    # print(align_hyp_3)

    ##############
    # STEP 5: Make sure that the reference consists of the same amount of words as the prompt.
    # The result of step 4 sees inserted words (in the hypothesis) as separate words in the ref (-------). 
    # To correct for this, we append the -------- to the next word. In this way, the prompt and hyp have the same length.
    # ------jo-jo met twee doppen v-an een fles- kun je bijvoorbeeld een jojo maken -------wat is een jojo nou weer vraagt teun dat is heel erg leuk -ouderwets speelgoed ---l-egt juf s-oets uit ze laat een jo--jo zien en doet voor hoe het werkt die jo--jo heeft ze zelf gemaakt van twee ro-de doppen daarna laat ze op het bord een -tekening zien daarop staat precies wat ze moeten doen om een jo--jo te maken dan legt ze heel veel gekleurde doppen op haar --tafel nu mogen jullie het allemaal zelf proberen -----zegt ze d-e klas is er- de hele middag ---druk mee bezig aan het eind- van de dag hebben ze ---samen wel dertig jojos gemaakt ik ben -echt ontzettend trots op jullie zegt de juf lachend daarom heb ik een leuke verrassing-- voor jullie ze zwaait naar iemand die op de gang staat de deur gaat open en meneer zemel -komt binnen hallo kinderen zegt hij vrolijk ik hoor van de juf dat jullie zo goed helpen met speelgoed maken hij loopt door de klas en bekijkt alle -jojos
    # begin jollo met twee donker voor een f-est kom je bijvoorbeeld een jong ma--n krijgt wat is een jong nou meer vraagt zeun dat is heel erg leuk nou------- --------- stilze-t j-e stoe-l uit ze laat een jongen zien en doet voor hoe het werkt d-e jongen heeft ze zelf gemaakt van twee ronde ------ ------ ---- -- -- --- ---- --- donke--n- ---- daarop staat precies wat ze moet-- doen om een jongen te maken dan zegt ze heel veel gekleurde donken op haar zetafel nu mogen juffie het allemaal zelf proberen zelf zegt ze dat k-an ik erg de hele middag de druk met de--- zin het einde van de dag h---et -- zilsamen --- ------ ----- ------- -- --- me--t ---------- ----- -- ----de ---- -- --- ------- ------ --- -- --- ----- ------jongen ---- ------ -- z---ij ---- ------ --- -- -- ---- ----- -- ---- ---- ---- -- ------ ----- maakt ------ ----- -------- ---- --- ------- -- ---- --- -- --- --- ------ -- ---- ------ --- --------- ----- --- ----- ---- -- ---- -- ------- ---- zilver
    #############

    align_ref_list = []
    align_hyp_list = []
    to_add_ref = ''
    to_add_hyp = ''

    for i,j in zip(align_ref_3, align_hyp_3):
            
            if len(i.replace('-', '')) != 0:
                to_add_ref += i
                to_add_hyp += j
                align_ref_list.append(to_add_ref)
                align_hyp_list.append(to_add_hyp)
                to_add_ref = ''
                to_add_hyp = ''

            else:
                to_add_ref += i + '-'
                to_add_hyp += j + ' '

    if to_add_ref != '' and to_add_hyp != '':
        align_ref_list[-1] += '-' + to_add_ref[:-1]
        align_hyp_list[-1] += ' ' + to_add_hyp[:-1]


    align_ref_4 = " ".join(align_ref_list)
    align_hyp_4 = " ".join(align_hyp_list)

    # print('version4')
    # print(align_ref_4)
    # print(align_hyp_4)

    return align_ref_4.replace('-', '*'), align_hyp_4.replace('-', '*')

def removeInsertionsAsterisk(s):
    return s.replace("*", "")

def trimPipesAndSpaces(s):
    return s.replace("|", " ").strip()

def two_way_alignment_modern(prompt, asrTrans):

    # Forward alignment
    prompt_align, asrTrans_align = one_way_alignment_modern(prompt, asrTrans)

    # REversed alignment
    # The reversed alignment part is removed, since this returned exactly the same as the forward alignment.

    align_ref_list, align_hyp_list = split_alignments_in_segments(prompt_align, asrTrans_align)
    
    # Create output DataFrame
    outputDF = pd.DataFrame()
    outputDF['prompt'] = pd.Series(align_ref_list).apply(removeInsertionsAsterisk).apply(trimPipesAndSpaces)
    outputDF['aligned_ref'] = pd.Series( align_ref_list).apply(trimPipesAndSpaces)
    outputDF['aligned_asrTrans'] = pd.Series(align_hyp_list).apply(trimPipesAndSpaces)
    outputDF['correct'] = outputDF.apply(determineCorrectness, axis=1)

    return outputDF



