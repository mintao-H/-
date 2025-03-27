import requests
api_key = 'SG5M4LUyFYidnSjYR'
city = 'nanning'
response = requests.get(f'https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={city}&language=zh-Hans&unit=c')
print(response)
result = response.json()
print(result)
txt_city = result['results'][0]['location']['name']
txt_wea = result['results'][0]['now']['text']
txt_tem = result['results'][0]['now']['temperature']
