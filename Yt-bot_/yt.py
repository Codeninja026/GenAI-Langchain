

import yt_dlp
import requests

video_url = 'https://www.youtube.com/watch?v=eMlx5fFNoYc'

def get_text_url(url,lang='en'):

    ydl_opts = {
        "skip_download":True,
        "writesubtitles":True,
        "writeautomaticsub":True,
        'subtitleslangs':[lang],
        'quiet':True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as yts:
        info = yts.extract_info(url,download=False)
        subs = info.get('subtitles') or info.get('automatic updates')

        if not subs or lang not in subs:
            raise ValueError(f"no sub found on lang {lang}")
        sub_url = subs[lang][0]['url']


        res = requests.get(sub_url)

        res.raise_for_status()
        data = res.json()  # parse JSON
        text_segments = []

        for event in data.get("events", []):
            if "segs" in event:
                for seg in event["segs"]:
                    text_segments.append(seg.get("utf8", ""))

        return " ".join(text_segments)






text = get_text_url(video_url)
print(text)