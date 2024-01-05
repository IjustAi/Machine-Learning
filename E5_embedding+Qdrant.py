from datasets import load_dataset
from qdrant_client import QdrantClient 
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams
from comet import download_model,load_from_checkpoint
from sentence_transformers import SentenceTransformer,util
from bleurt import score
from tqdm import tqdm
import time 
import datasets
import openai
import nltk
import pandas as pd 

def embedding_score(df,Qdrant_Api_Key,Qdrant_Url,collection_name):
    qdrant_client = QdrantClient(
        url = Qdrant_Url,api_key=Qdrant_Api_Key,timeout=6000, prefer_grpc=True
    )

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
   
    zh = df['ZH'].tolist()
    batch_size = 100
    total_batches = len(zh) // batch_size + 1

    embeddings = []

    for i in tqdm(range(total_batches)):
        start_idx=i*batch_size
        end_idx = min((i+1)*batch_size,len(zh))
        batch = zh[start_idx:end_idx]
        batch_embbedings = embedding_model.encode(batch,normalize_embeddings=True)
        embeddings.extend(batch_embbedings)
    
    embedding_list = [embedding.tolist() for embedding in embeddings]

    temp_df = df.copy()
    temp_df['embeddings'] =embedding_list
    temp_df['id'] = temp_df.index
    tqdm.pandas()

    points = temp_df.progress_apply(
        lambda row:PointStruct(
            id=row['id'],
            vector=row['embeddings'],
            payload={
                'ZH':row['ZH'],
                'EN':row['EN']
            },
        ),
        axis=1,
    ).tolist()

    info= qdrant_client.upsert(
        collection_name=collection_name,wait=True,points=points
    )

    return points

def get_icl_examples(user_prompt,collection_name,Qdrant_Api_Key,Qdrant_Url):
    qdrant_client = QdrantClient(
        url = Qdrant_Url,api_key=Qdrant_Api_Key,timeout=6000, prefer_grpc=True
    )

    embedding_model =SentenceTransformer('intfloat/multilingual-e5-large')
    embeddings =list(embedding_model.encode([user_prompt],normalize_embeddings=True))
    query_embedding = embeddings[0].tolist()

    num_of_retrieve=5

    res= qdrant_client.search(
        collection_name = collection_name,
        query_vector =query_embedding,
        with_payload=True,
        limit = num_of_retrieve,
    )

    ICL_examples = {}
    for item in res:
        zh_text=item.payload['ZH']
        en_text=item.payload['EN']
        ICL_examples[zh_text] = en_text

    return ICL_examples

def translate_text(text,example_translations):
    openai.api_type = ""
    openai.api_base = ""
    openai.api_version = ""
    openai.api_key = ""

    example_translations_str = "\n".join([f"{chinese_text}: {english_text}" for chinese_text, english_text in example_translations.items()])

    messages = [
        {"role": "system", "content": "You are a translation assistant from Chinese to English. Some rules to remember:\n\n- Do not add extra blank lines.\n- It is important to maintain the accuracy of the contents, but we don't want the output to read like it's been translated. So instead of translating word by word, prioritize naturalness and ease of communication.\n\n Here are some examples that you can use to learn how to translate from Chinese to English:\n"+ example_translations_str},
        {"role": "user", "content":f'Please translate the given Chinese sentence {text} to English sentence and please make the translation as accurate and natural as possible.' }
    ]
    try:
        response = openai.ChatCompletion.create(
            engine=" ",
            messages = messages,
            temperature=0.7,
            max_tokens=800,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=None)
        translation= response['choices'][0]['message']['content']

    except KeyError:
        translation = ''

    return translation

def calculate_BLEU(reference_translations, translations):
    bleu_scores = []
    for reference, pre in zip(reference_translations, translations):
        reference_tokens = nltk.word_tokenize(reference)
        pre_tokens = nltk.word_tokenize(pre)

        if not reference_tokens or not pre_tokens:
            continue

        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            [reference_tokens], pre_tokens,
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2
        )
        bleu_scores.append(bleu_score)

    average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return average_bleu_score

def calculate_COMET(source_sentences,translations,reference_translations):
    model_path=download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data=[]

    for  src, pre,reference in zip(source_sentences,translations,reference_translations):
        data.append({

            "src":src,
            "mt":pre,
            "ref":reference
        })
    model_output = model.predict(data, batch_size=8, gpus=0)
    return model_output

def calculate_BLEURT(reference_translations,translations):
    checkpoint = r"/Users/chenyufeng/bleurt/bleurt/BLEURT-20"
    scorer = score.BleurtScorer( checkpoint)
    scores = scorer.score(references=reference_translations, candidates=translations)
    total_score = sum(scores)/len(scores)
    return total_score

def main():
    Qdrant_Api_Key=''
    Qdrant_Url = ''

    dataset= datasets.load_dataset('opus100','en-zh',split='train[0:1000]')
    ZH = [item['translation']['zh'] for item in dataset] 
    EN = [item['translation']['en'] for item in dataset]

    df = pd.DataFrame({'ZH':ZH,'EN':EN})
    collection_name = "E5embedding"

    points = embedding_score(df,Qdrant_Api_Key,Qdrant_Url, collection_name)

    opus_test = datasets.load_dataset('opus100','en-zh',split='test[0:100]')
    source_sentences = [item['translation']['zh'] for item in opus_test]
    reference_translations= [item['translation']['en'] for item in opus_test]

    translations = []

    for index in range(len(source_sentences)):
        zh = source_sentences[index]
        example_translations=get_icl_examples(zh,collection_name,Qdrant_Api_Key,Qdrant_Url)
        translation = translate_text(zh,example_translations)
        if translation == "":
            continue
        else:
            translations.append(translation)

        print(f'finished {index+1}')
        time.sleep(2)

    average_bleu_score = calculate_BLEU(reference_translations, translations)
    print("BLEU score:", average_bleu_score)

    comet_score = calculate_COMET(source_sentences,translations,reference_translations)
    print("COMET score:", comet_score) 

    bleurt_score = calculate_BLEURT(reference_translations,translations)
    print("BLEURT score:", bleurt_score) 

if __name__ =="__main__":
    main()
