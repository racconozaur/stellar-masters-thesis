import requests
import json
import csv
import os

BASE_URL = "https://api.stellar.expert"
DIRECTORY_URL = f"{BASE_URL}/explorer/public/directory?limit=200"

DIRECTORY_JSON_FILE = "full_stellar_directory.json"
DIRECTORY_CSV_FILE = "full_stellar_directory.csv"

def fetch_full_directory():
    url = DIRECTORY_URL
    headers = {"accept": "application/json"}
    all_records = []

    while url:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {url}, Status code: {response.status_code}")
            break

        data = response.json()
        records = data.get('_embedded', {}).get('records', [])
        all_records.extend(records)

        # pagination
        next_link = data.get('_links', {}).get('next', {}).get('href')
        if next_link:
            url = BASE_URL + next_link
        else:
            url = None

    print(f"Total directory entries fetched: {len(all_records)}")
    return all_records

def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to {filename}")

def save_to_csv(data, filename):
    if not data:
        return

    fieldnames = ['organization_name', 'description', 'domains', 'tags', 'rating', 'account_address']

    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in data:
            name = entry.get('name', '')
            description = entry.get('description', '')
            domains = ', '.join(entry.get('domains', []))
            tags = ', '.join(entry.get('tags', []))
            rating = entry.get('rating', '')
            accounts = entry.get('accounts', [])

            for account in accounts:
                writer.writerow({
                    'organization_name': name,
                    'description': description,
                    'domains': domains,
                    'tags': tags,
                    'rating': rating,
                    'account_address': account
                })

    print(f"Saved CSV to {filename}")

def main():
    directory_data = fetch_full_directory()

    save_to_json(directory_data, DIRECTORY_JSON_FILE)
    save_to_csv(directory_data, DIRECTORY_CSV_FILE)

if __name__ == "__main__":
    main()
