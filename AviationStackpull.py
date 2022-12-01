import requests
import csv
params = {
  'access_key': 'b0441f84a9fa72e44c87967db8eb32a6',
  'dep_iata':'LAX',
  'flight_status':'landed'
}
with open('newLAXFlightData11-12-2022.csv', 'w',newline='') as ofile:
    writer = csv.writer(ofile, delimiter=',')
    writer.writerow(['dep airport', 'dep delay','arr airport','arr delay'])
    

    
    try:
        api_result = requests.get('http://api.aviationstack.com/v1/flights', params)
    except requests.exceptions.ConnectionError:
       print("Connect Failed")
    api_response = api_result.json()
    print(api_response)
    for flight in api_response['data']:
        writer.writerow([
            flight['departure']['iata'],
            flight['departure']['delay'],
            flight['arrival']['iata'],
            flight['arrival']['delay']])