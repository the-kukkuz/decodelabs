import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TechStackRecommender:
    """
    Digital Matchmaker Engine: Maps raw user skills to job roles 
    using TF-IDF weighting and Cosine Similarity.
    """
    
    def __init__(self, data):
        # 1. Initialize dataset
        self.df = pd.DataFrame(data)
        
        # Format skills into a single string for text processing
        self.df['skills_text'] = self.df['skills'].apply(lambda x: ' '.join(x).lower())
        
        # 2. Vector Mapping using TF-IDF (Slide 11 & 12)
        # This penalizes generic words and rewards highly specific technical skills
        self.vectorizer = TfidfVectorizer()
        self.item_vectors = self.vectorizer.fit_transform(self.df['skills_text'])

    def get_recommendations(self, user_skills, top_n=3):
        """
        Executes the 4-step pipeline: Ingestion -> Scoring -> Sorting -> Filtering
        """
        # STEP 1: Ingestion (Format user input)
        user_text = ' '.join(user_skills).lower()
        user_vector = self.vectorizer.transform([user_text])
        
        # STEP 2: Scoring (Calculate Cosine Similarity - Slide 15)
        # Measures the angle/alignment between user profile and job roles
        similarity_scores = cosine_similarity(user_vector, self.item_vectors).flatten()
        
        # Add scores to the dataframe for sorting
        self.df['match_score'] = similarity_scores
        
        # STEP 3: Sorting (Descending order)
        sorted_df = self.df.sort_values(by='match_score', ascending=False)
        
        # STEP 4: Filtering (Prevent choice overload by returning Top-N)
        top_matches = sorted_df.head(top_n)
        
        return top_matches[['role', 'match_score']]


def main():
    # --- MOCK DATASET (Simulating 'raw_skills.csv' from Slide 22) ---
    job_data = [
        {"role": "Data Scientist", "skills": ["python", "sql", "machine learning", "data analysis", "pandas"]},
        {"role": "DevOps Engineer", "skills": ["aws", "docker", "kubernetes", "linux", "ci/cd", "python"]},
        {"role": "Backend Developer", "skills": ["java", "python", "sql", "apis", "node.js", "databases"]},
        {"role": "Frontend Developer", "skills": ["javascript", "html", "css", "react", "vue", "web design"]},
        {"role": "Cloud Architect", "skills": ["aws", "azure", "gcp", "security", "networking", "infrastructure"]},
        {"role": "Mobile App Developer", "skills": ["swift", "kotlin", "ios", "android", "flutter", "react native"]}
    ]

    # Initialize the Recommendation Engine
    engine = TechStackRecommender(job_data)
    
    print("="*50)
    print("🚀 DECODELABS DIGITAL MATCHMAKER INITIATED")
    print("Project 3: Tech Stack Recommender")
    print("="*50)
    
    while True:
        print("\nType 'exit' to quit the engine.")
        # Step 1 Requirement: Accept a minimum of 3 inputs (Slide 18)
        user_input = input("Enter your skills separated by commas (min. 3 required):\n> ")
        
        if user_input.strip().lower() == 'exit':
            print("Shutting down the matchmaker. Goodbye!")
            break
            
        # Clean input
        skills_list = [skill.strip() for skill in user_input.split(',')]
        
        # Enforce Minimum Data Density (Anti-Cold Start)
        if len(skills_list) < 3 and skills_list[0] != '':
            print("⚠️ Insufficient Data: Please enter at least 3 skills for accurate pattern alignment.")
            continue
        elif skills_list[0] == '':
            continue
            
        # Fetch Recommendations
        print("\n⚙️ Calculating Vector Alignment...")
        recommendations = engine.get_recommendations(skills_list, top_n=3)
        
        # Display Output
        print("\n🎯 TOP 3 RECOMMENDED CAREER PATHS:")
        print("-" * 40)
        
        # Check if the highest score is 0 (Vocabulary mismatch)
        if recommendations.iloc[0]['match_score'] == 0:
             print("No mathematical alignment found. Try using industry-standard technical terms.")
        else:
            for index, row in recommendations.iterrows():
                # Convert decimal score to percentage for intuitive reading (Slide 16)
                match_percentage = round(row['match_score'] * 100, 1)
                print(f"➜ {row['role'].ljust(25)} | Match: {match_percentage}%")
        print("-" * 40)

if __name__ == "__main__":
    main()