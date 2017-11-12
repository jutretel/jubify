import os
import pandas as pd
import urllib
import lxml.html
import pyprind

class Song(object):
    def __init__(self, artist, title):
        self.artist = self.__format_str(artist)
        self.title = self.__format_str(title)
        self.url = None
        self.lyric = None

    def __format_str(self, s):
        # remove paranthesis and contents
        s = s.strip()
        try:
            # strip accent
            s = ''.join(c for c in unicodedata.normalize('NFD', s)
                         if unicodedata.category(c) != 'Mn')
        except:
            pass
        s = s.title()
        return s

    def __quote(self, s):
         return urllib.parse.quote(s.replace(' ', '_'))

    def __make_url(self):
        artist = self.__quote(self.artist)
        title = self.__quote(self.title)
        artist_title = '%s:%s' %(artist, title)
        url = 'http://lyrics.wikia.com/' + artist_title
        self.url = url

    def update(self, artist=None, title=None):
        if artist:
            self.artist = self.__format_str(artist)
        if title:
            self.title = self.__format_str(title)

    def lyricwikia(self):
        self.__make_url()
        try:
            doc = lxml.html.parse(self.url)
            lyricbox = doc.getroot().cssselect('.lyricbox')[0]
        except (IOError, IndexError) as e:
            self.lyric = ''
            return self.lyric
        lyrics = []

        for node in lyricbox:
            if node.tag == 'br':
                lyrics.append('\n')
            if node.tail is not None:
                lyrics.append(node.tail)
        self.lyric =  "".join(lyrics).strip()
        return self.lyric


def make_artist_table(base):

# Get file names

    files = [os.path.join(base,fn) for fn in os.listdir(base) if fn.endswith('.h5')]
    data = {'file':[], 'artist':[], 'title':[]}

    # Add artist and title data to dictionary
    for f in files:
        store = pd.HDFStore(f)
        title = store.root.metadata.songs.cols.title[0]
        artist = store.root.metadata.songs.cols.artist_name[0]
        data['file'].append(os.path.basename(f))
        data['title'].append(title.decode("utf-8"))
        data['artist'].append(artist.decode("utf-8"))
        store.close()

    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='columns')
    df = df[['file', 'artist', 'title']]
    return df

base = './MillionSongSubset/'
df = make_artist_table(base)

df['lyrics'] = pd.Series('', index=df.index)
df.tail()

pbar = pyprind.ProgBar(df.shape[0])
for row_id in df.index:
    song = Song(artist=df.loc[row_id]['artist'], title=df.loc[row_id]['title'])
    lyr = song.lyricwikia()
    df.loc[row_id,'lyrics'] = lyr
    pbar.update()

df.to_csv('./df_lyr_backup.csv')
