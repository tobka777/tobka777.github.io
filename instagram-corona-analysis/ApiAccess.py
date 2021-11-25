import requests
from datetime import datetime
import json

data = []
r = requests.get("https://www.instagram.com/explore/tags/coronavirus/?__a=1")
roh = r.json()['graphql']
elem = roh['hashtag']['edge_hashtag_to_media']['edges']
i = 0
time = 1577836800 #2020-01-01
for e in elem:
    d = e['node']
    #print(datetime.utcfromtimestamp(d['taken_at_timestamp']).strftime('%Y-%m-%d'), d["id"])
    if len(e['node']['edge_media_to_caption']['edges']) > 0 and d['taken_at_timestamp'] > time:
        d['text'] = e['node']['edge_media_to_caption']['edges'][0]['node']['text']
        data.extend([d])
        i += 1

count = str(4)
breakCond = False
end_cursor = []
if roh['hashtag']['edge_hashtag_to_media']['page_info']['has_next_page']:
    next = roh['hashtag']['edge_hashtag_to_media']['page_info']['end_cursor']

    while breakCond == False and roh['hashtag']['edge_hashtag_to_media']['page_info']['has_next_page']:
        r = requests.get('https://www.instagram.com/explore/tags/coronavirus/?__a=1&max_id='+next)
        if r.status_code == 200:
            roh = r.json()['graphql']
            elem = roh['hashtag']['edge_hashtag_to_media']['edges']
            lasttime = 0
            for e in elem:
                d = e['node']
                #print(datetime.utcfromtimestamp(d['taken_at_timestamp']).strftime('%Y-%m-%d'), d["id"])
                if len(e['node']['edge_media_to_caption']['edges']) > 0 and d['taken_at_timestamp'] > time:
                    d['text'] = e['node']['edge_media_to_caption']['edges'][0]['node']['text']
                    data.extend([d])
                    i += 1
                lasttime = int(d['taken_at_timestamp'])

            if i > 1000000:
                breakCond = True
            print(datetime.utcfromtimestamp(lasttime).strftime('%Y-%m-%d'), i)

            if roh['hashtag']['edge_hashtag_to_media']['page_info']['has_next_page']:
                next = roh['hashtag']['edge_hashtag_to_media']['page_info']['end_cursor']
                end_cursor.append(next)
            else:
                breakCond = True

            if (i % 10000) < 69:
                with open('data_'+str(i)+'.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                with open('cursor_'+str(i)+'.json', 'w', encoding='utf-8') as f:
                    json.dump(end_cursor, f, ensure_ascii=False, indent=4)
        else:
            print(r.status_code)

print(i)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

with open('cursor.json', 'w', encoding='utf-8') as f:
    json.dump(end_cursor, f, ensure_ascii=False, indent=4)