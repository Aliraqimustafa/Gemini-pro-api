from pyngrok import ngrok

def kill(kill_input):
    if kill_input:
         [ngrok.disconnect(url.public_url) for url in ngrok.get_tunnels()]
        # print('Disconnection completed')
    pass

kill('y')


__PORT__ = 7860
public_url = ngrok.connect(__PORT__).public_url
print(f'\n\n\n Now you can send post request : {public_url}\n\n\n')

import time
time.sleep(60*100)
