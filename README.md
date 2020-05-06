# line-bot-demo

1. Copy config file
   ```
   $ cp config_template.yml config.yml
   ```

2. Add your API token and secret key to config.yml
   ![](materials/token.png)
   
   ![](materials/secret.png)
   ```
   $ vim config.yml
   ```
   
   ```
   Linebot:
    access_token: <CHANNEL ACCESS TOKE>
    secret: <CHANNEL SECRET>
   ```
   
2. Run app.py
   ```
   $ python app.py
   ```
   
   ![](materials/app.png)

3. [option] Local host as a server
    1. Download [ngrok](https://ngrok.com/download) to 'line-bot-demo/ngrok'
    2. Run local_server.sh
       ```
       $ chmod +x local_server.sh
       $ ./local_server.sh
        ```
    3. Copy url to webhook to LINE developers
       ![](materials/ngrok.png)
       
       ![](materials/update_webhook.png)
       
4. Echo bot demo
    
    ![](materials/demo.png)
   