import json
import requests
import pandas as pd

from config import *
 
def query_sentiment_trends(search):
    query = {
        "size": 0,
        "aggs": {
            "group_by_date": {
            "date_histogram": {
                "field": "created_at",
                "interval": "day"
            },
            "aggs": {
                "group_by_sentiment": {
                "terms": {
                    "field": "sentiment.keyword"
                }
                }
            }
            }
        }
    }

    if search:
        query['query'] = {
            'match': {
                'title': search
            }
        }

    query = json.dumps(query)

    headers = {
        'Content-Type': 'application/json'
    }

    resp = requests.get(
        f"{ELASTICSEARCH_URL}/news/_search",
        headers=headers,
        data=query,
        auth=ELASTICSEARCH_AUTH,
    )

    print(f'Status code: {resp.status_code}')

    results = resp.json()
    buckets = results['aggregations']['group_by_date']['buckets']

    buffer = [] 
    for x in buckets:
        print(json.dumps(x, indent=2))

        sents = x['group_by_sentiment']['buckets']

        entry = {t['key']: t['doc_count'] for t in sents}
        entry['date'] = x['key_as_string']

        buffer.append(entry)

    if len(buffer) == 0:
        return []

    df = pd.DataFrame(buffer)
    df['date'] = df['date'].str[:10]
    df = df.set_index('date')
    df = df.fillna(0)

    if 'positive' not in df.columns:
        df['positive'] = 0

    if 'negative' not in df.columns:
        df['negative'] = 0

    df['sentiment'] = (df['positive'] - df['negative']) / df.sum(axis=1)
    df = df.dropna()
    df = df.reset_index()

    return df.to_dict(orient='records')

def main(event, context):
    params = event.get('queryStringParameters')

    if params:
        search = params.get('search')
    else:
        search = None

    trends = query_sentiment_trends(search)

    body = {
        'trends': trends,
    }

    print('body')

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        # CORS
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": True,
        }
    }

    return response