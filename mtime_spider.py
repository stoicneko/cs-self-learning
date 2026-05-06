# -*- coding: utf-8 -*-

import random
import time
import pandas as pd
import requests
import jieba
from gensim.models import LdaModel
from gensim.corpora import Dictionary


# ======================
# 第一部分：数据爬取
# ======================
def mtime_spider(movie_id, max_comments=200):
    headers = {
        'Cookie': 'Hm_lvt_07aa95427da600fc217b1133c1e84e5b=1778039614; HMACCOUNT=E3C22092ADAE6441; searchHistoryCookie=%u96C4%u72EE%2C%u96C4%u72EE%u5C11%u5E74; smidV2=202605052054170613a9185ce3888cf4b52cbf4b2e1bc200b6b0de93c0b9a90; Hm_lpvt_07aa95427da600fc217b1133c1e84e5b=1778039658; .thumbcache_c835bd016c362d09f235c52c6fca20c7=ZlagUPbIxIuWMcKA5bw6mBeJm5jsE5PB8j6s/NfATY4pc4KF2pVIHzH/23PXTskBr+nTrCKbOE0g4OMW8tbauw%3D%3D',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0'
    }

    comments = []
    page_index = 1
    order_type = 1

    while len(comments) < max_comments:
        timestamp = int(time.time() * 1000)
        url = (
            f'https://front-gateway.mtime.com/library/movie/comment.api?'
            f'tt={timestamp}&movieId={movie_id}&pageIndex={page_index}&pageSize=20&orderType={order_type}'
        )
        try:
            time.sleep(random.uniform(2, 4))
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            comment_list = data.get('data', {}).get('list', [])
            if not comment_list:
                break

            for item in comment_list:
                user = item.get('nickname', '').strip() or '匿名用户'
                comment = item.get('content', '').strip()
                if comment and len(comments) < max_comments:
                    comments.append({'user': user, 'comment': comment})

            page_index += 1

        except Exception:
            break

    df = pd.DataFrame(comments[:max_comments])
    df.to_csv('Mtime comments_原始数据.csv', index=False, encoding='utf-8-sig')
    print(f"爬取完成：共获取 {len(df)} 条有效评论（目标 {max_comments} 条）")
    return df


# ======================
# 第二部分：数据预处理
# ======================
def preprocess_comments(df):
    stopwords = set()
    try:
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        pass

    processed_comments = []
    for comment in df['comment']:
        words = jieba.lcut(comment)
        filtered = [
            word for word in words
            if word not in stopwords
               and len(word) > 1
               and not word.isspace()
               and not word.isdigit()
        ]
        processed_comments.append(filtered)

    return processed_comments


# ======================
# 第三部分：LDA主题建模
# ======================
def lda_analysis(processed_docs, num_topics=5):
    dictionary = Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=3, no_above=0.6)

    corpus = [dictionary.doc2bow(text) for text in processed_docs if text]
    if not corpus:
        raise ValueError("语料库为空")

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto'
    )
    return lda_model, dictionary


# ======================
# 主程序
# ======================
if __name__ == '__main__':
    MOVIE_ID = '270067'  # 雄狮少年
    MAX_COMMENTS = 200

    print("=== 开始爬取时光网影评 ===")
    df = mtime_spider(MOVIE_ID, MAX_COMMENTS)

    print("=== 开始数据分析 ===")
    processed = preprocess_comments(df)

    try:
        lda_model, dictionary = lda_analysis(processed)

        keywords_list = []
        weights_list = []
        for doc in processed:
            if not doc:
                keywords_list.append('')
                weights_list.append('')
                continue

            bow = dictionary.doc2bow(doc)
            topics = lda_model.get_document_topics(bow)
            if not topics:
                keywords_list.append('')
                weights_list.append('')
                continue

            main_topic = max(topics, key=lambda x: x[1])
            topic_words = lda_model.show_topic(main_topic[0], topn=5)
            keywords = [word for word, _ in topic_words]
            weights = [f"{weight:.4f}" for _, weight in topic_words]

            keywords_list.append(', '.join(keywords))
            weights_list.append(', '.join(weights))

        df['评论关键词'] = keywords_list
        df['关键词权重'] = weights_list
        df.to_excel('Mtime comments.xlsx', index=False, engine='openpyxl')
        print("=== 分析完成：结果已保存至 Mtime comments.xlsx ===")

    except ValueError:
        print("=== 分析失败：有效评论数据不足 ===")
