import pandas as pd
import os
import json
import time
import random
import requests

# Set a placeholder for the API key which will be automatically provided by the environment
API_KEY = "" #fill in your own api
# Use the Gemini API endpoint for text generation
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

def generate_synthetic_reviews(prompt: str, num_examples: int, headers: dict) -> list:
    """
    Generates synthetic reviews using a language model with a structured JSON output.
    Uses exponential backoff for API calls to handle rate limiting.
    """
    reviews = []
    retries = 3
    delay = 1

    # The JSON schema defines the format of the generated output
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "review_text": {"type": "STRING"},
                "violation_type": {"type": "STRING"}
            },
            "required": ["review_text", "violation_type"]
        }
    }

    # The payload for the API call requests a specific JSON format
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    }

    while len(reviews) < num_examples:
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{API_URL}?key={API_KEY}",
                    headers=headers,
                    data=json.dumps(payload)
                )
                response.raise_for_status()
                response_data = response.json()
                
                # Safely parse the JSON response from the model
                candidate = response_data.get('candidates', [{}])[0]
                content = candidate.get('content', {}).get('parts', [{}])[0].get('text', '{}')
                generated_json = json.loads(content)
                
                if isinstance(generated_json, list):
                    # Filter out any malformed entries and add to our list
                    for entry in generated_json:
                        if all(k in entry for k in ["review_text", "violation_type"]):
                            reviews.append(entry)
                            if len(reviews) >= num_examples:
                                break
                break  # Exit retry loop on success

            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    print("Max retries reached. Exiting API call.")
                    return reviews
    return reviews


def main():
    """
    Main function to orchestrate the data augmentation process.
    """
    # Define file paths
    manually_labeled_csv = '../data/manual_labeled_reviews_200.csv'
    output_csv = '../data/augmented_labeled_reviews.csv'
    
    # Check if the input file exists
    if not os.path.exists(manually_labeled_csv):
        print(f"Error: The input file '{manually_labeled_csv}' was not found.")
        return
        
    try:
        # Load the manually labeled dataset
        labeled_df = pd.read_csv(manually_labeled_csv)
        
        # Define categories and the number of examples to generate for each
        categories_to_augment = {
            "1": {"name": "Advertisement", "count": 200},
            "2": {"name": "Irrelevant Content", "count": 200},
            "3": {"name": "Rant Without Visit", "count": 200}
        }
        
        # Use a single review entry as a template for all synthetic data
        template_entry = {
            'text': 'Great food and customer service is A+.',
            'rating': 5,
            'time': 1495245139184,
            'user_id': 101845303640982410214,
            'gmap_id': '0x80db512222e6feff:0x2c42561d7e32661e',
            'category': "['Hamburger restaurant', 'Fast food restaurant']",
            'description': 'No-nonsense counter serve offering ample burger meals & global eats, including gyros & Mexican fare.',
            'avg_rating': 4.5,
            'num_of_reviews': 853,
            'violation_type': '4'
        }
        
        synthetic_reviews = []
        headers = {'Content-Type': 'application/json'}
        
        print("\n--- Starting LLM-Powered Data Augmentation ---")
        for label, cat_info in categories_to_augment.items():
            category_name = cat_info['name']
            num_to_generate = cat_info['count']
            
            print(f"Generating {num_to_generate} examples for category: {category_name} ({label})...")
            
            # Create a DataFrame by replicating the template entry
            template_df = pd.DataFrame([template_entry] * num_to_generate)
            
            # --- Craft the prompt based on the category ---
            if category_name == "Irrelevant Content":
                # Use the template's category for context
                prompt_text = (
                    f"Generate a list of {num_to_generate} fake business reviews that are "
                    f"completely irrelevant to a fast food restaurant. The reviews should not "
                    f"contain any mention of the business, its products, or its services. "
                    f"Each review must be a single JSON object with 'review_text' and 'violation_type'='2'."
                    f"\n\nExample Reviews:"
                    f"\n- Review: 'I'm so excited about the new movie that's coming out next month!'"
                    f"\n- Review: 'My car broke down on the freeway today, what a terrible experience!'"
                    f"\n- Review: 'Does anyone have a good recipe for a homemade smoothie? I need inspiration.'"
                )
            elif category_name == "Advertisement":
                prompt_text = (
                    f"Generate a list of {num_to_generate} fake business reviews that are "
                    f"clearly advertisements. The reviews should contain promotional language, "
                    f"links to external websites, or phone numbers. "
                    f"Each review must be a single JSON object with 'review_text' and 'violation_type'='1'."
                    f"\n\nExample Reviews:"
                    f"\n- Review: 'This place is great! Visit www.mygreatdeals.com for a coupon!'"
                    f"\n- Review: 'For a great mortgage rate, call us at 555-123-4567!'"
                    f"\n- Review: 'This place is a scam, but you can get a free ebook by clicking here: bit.ly/free-ebook'"
                )
            else: # Rant Without Visit
                prompt_text = (
                    f"Generate a list of {num_to_generate} fake business reviews that are "
                    f"rants written by someone who never visited the location. They should "
                    f"reference prices, rumors, or secondhand information. "
                    f"Each review must be a single JSON object with 'review_text' and 'violation_type'='3'."
                    f"\n\nExample Reviews:"
                    f"\n- Review: 'I heard from a friend that the prices here are way too high. I'm not going.'"
                    f"\n- Review: 'The owner is so rude according to the local gossip. I can't believe they're still in business.'"
                    f"\n- Review: 'My neighbor told me they never wash the dishes. Absolutely disgusting!'"
                )

            # Generate new text from the LLM
            generated_list = generate_synthetic_reviews(prompt_text, num_to_generate, headers)
            generated_df = pd.DataFrame(generated_list)
            
            # Merge the new text with the existing template rows
            if not generated_df.empty:
                template_df['text'] = generated_df['review_text']
                template_df['violation_type'] = generated_df['violation_type']
                synthetic_reviews.append(template_df)

            print(f"Generated {len(generated_list)} examples for {category_name}.")

        # Combine all dataframes
        synthetic_df = None
        if synthetic_reviews:
            synthetic_df = pd.concat(synthetic_reviews, ignore_index=True)
            final_df = pd.concat([labeled_df, synthetic_df], ignore_index=True)
        else:
            final_df = labeled_df
        
        # Save the final augmented dataset
        final_df.to_csv(output_csv, index=False)
        
        print(f"\nOriginal labeled dataset size: {len(labeled_df)}")
        print(f"Synthetic examples added: {len(synthetic_df) if synthetic_df is not None else 0}")
        print(f"Final augmented dataset size: {len(final_df)}")
        print(f"Augmented dataset saved to '{output_csv}'.")
        print("\nDistribution of labels in the new dataset:")
        print(final_df['violation_type'].value_counts())

    except Exception as e:
        print(f"An error occurred during augmentation: {e}")

if __name__ == "__main__":
    main()
