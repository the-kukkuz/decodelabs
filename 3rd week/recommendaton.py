import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class RecommendationEngine:
    """
    AI Recommendation Logic System
    Uses TF-IDF vectorization and Cosine Similarity for pattern matching
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.item_data = None
        self.item_vectors = None
        self.user_profile = []
        
    def load_sample_data(self):
        """Load predefined item database"""
        # Sample dataset: Tech Learning Paths
        data = {
            'item_id': range(1, 21),
            'title': [
                'Python Programming Fundamentals',
                'Machine Learning with Python',
                'Web Development with React',
                'Data Science Essentials',
                'Cloud Computing on AWS',
                'Deep Learning Neural Networks',
                'JavaScript for Beginners',
                'SQL Database Management',
                'DevOps and CI/CD Pipeline',
                'Artificial Intelligence Basics',
                'Cybersecurity Fundamentals',
                'Mobile App Development',
                'Blockchain Technology',
                'Natural Language Processing',
                'Computer Vision Projects',
                'Backend Development with Node.js',
                'Frontend Design with CSS',
                'Big Data Analytics',
                'IoT and Embedded Systems',
                'Quantum Computing Introduction'
            ],
            'description': [
                'Learn Python basics, variables, loops, functions, and object-oriented programming',
                'Build ML models using scikit-learn, pandas, numpy for predictive analytics',
                'Create modern web applications using React, JSX, hooks, and state management',
                'Master data analysis, visualization, statistics, and exploratory data techniques',
                'Deploy scalable applications on AWS using EC2, S3, Lambda, and cloud services',
                'Understand neural networks, backpropagation, CNNs, RNNs for deep learning',
                'JavaScript fundamentals, ES6 features, DOM manipulation, async programming',
                'Design relational databases, write SQL queries, optimize database performance',
                'Automate deployment with Docker, Kubernetes, Jenkins, and DevOps practices',
                'Explore AI concepts, algorithms, machine learning, and intelligent systems',
                'Learn security principles, encryption, network security, ethical hacking basics',
                'Build iOS and Android apps using Flutter, React Native, or native frameworks',
                'Understand blockchain, cryptocurrencies, smart contracts, and decentralized apps',
                'Process human language with NLP, sentiment analysis, chatbots, transformers',
                'Implement image recognition, object detection, face recognition using OpenCV',
                'Build REST APIs, server-side logic, authentication with Node.js and Express',
                'Master responsive design, animations, Flexbox, Grid, modern CSS techniques',
                'Process large datasets using Hadoop, Spark, distributed computing frameworks',
                'Connect physical devices, sensors, microcontrollers, and IoT protocols',
                'Introduction to quantum mechanics, qubits, quantum algorithms, and computing'
            ],
            'category': [
                'Programming', 'AI/ML', 'Web Development', 'Data Science', 'Cloud',
                'AI/ML', 'Programming', 'Database', 'DevOps', 'AI/ML',
                'Security', 'Mobile', 'Blockchain', 'AI/ML', 'AI/ML',
                'Web Development', 'Web Development', 'Data Science', 'IoT', 'Emerging Tech'
            ]
        }
        
        self.item_data = pd.DataFrame(data)
        self.item_data['combined_features'] = (
            self.item_data['title'] + ' ' + 
            self.item_data['description'] + ' ' + 
            self.item_data['category']
        )
        
        # Create TF-IDF vectors for all items
        self.item_vectors = self.vectorizer.fit_transform(
            self.item_data['combined_features']
        )
        
    def display_items(self, items_df, title="Available Items"):
        """Display items in a clean format"""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print('='*80)
        
        for idx, row in items_df.iterrows():
            print(f"\n[{row['item_id']}] {row['title']}")
            print(f"    Category: {row['category']}")
            print(f"    {row['description'][:100]}...")
            
        print('='*80)
    
    def get_user_preferences(self):
        """Capture user interests"""
        print("\n" + "="*80)
        print("STEP 1: BUILD YOUR PROFILE".center(80))
        print("="*80)
        
        print("\nDescribe your interests, skills, or what you want to learn.")
        print("Examples: 'machine learning', 'web design', 'cloud computing'")
        print("\nEnter at least 3 interests (comma-separated):")
        
        user_input = input("\n> ").strip()
        
        if not user_input:
            return False
            
        self.user_profile = [item.strip() for item in user_input.split(',')]
        
        if len(self.user_profile) < 3:
            print("\n⚠️  Please enter at least 3 interests for better recommendations.")
            return False
            
        print(f"\n✓ Profile created with {len(self.user_profile)} interests")
        return True
    
    def calculate_similarity(self):
        """Calculate cosine similarity between user profile and items"""
        # Combine user interests into a single text
        user_text = ' '.join(self.user_profile)
        
        # Transform user profile to TF-IDF vector
        user_vector = self.vectorizer.transform([user_text])
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(user_vector, self.item_vectors).flatten()
        
        return similarity_scores
    
    def get_recommendations(self, top_n=5):
        """Generate top N recommendations"""
        print("\n" + "="*80)
        print("STEP 2: PATTERN ALIGNMENT IN PROGRESS...".center(80))
        print("="*80)
        
        # Calculate similarity scores
        scores = self.calculate_similarity()
        
        # Add scores to dataframe
        self.item_data['similarity_score'] = scores
        
        # Sort by similarity score
        recommendations = self.item_data.sort_values(
            by='similarity_score', 
            ascending=False
        ).head(top_n)
        
        return recommendations
    
    def display_recommendations(self, recommendations):
        """Display recommendations with match scores"""
        print("\n" + "="*80)
        print("🎯 YOUR PERSONALIZED RECOMMENDATIONS".center(80))
        print("="*80)
        
        if recommendations['similarity_score'].iloc[0] == 0:
            print("\n⚠️  No strong matches found.")
            print("Try using different keywords or more specific interests.")
            return
        
        print("\nBased on your profile, here are your top matches:\n")
        
        for idx, row in recommendations.iterrows():
            match_percentage = row['similarity_score'] * 100
            
            # Skip items with 0% match
            if match_percentage == 0:
                continue
                
            print(f"{'─'*80}")
            print(f"Rank #{recommendations.index.get_loc(idx) + 1} | "
                  f"Match: {match_percentage:.1f}%")
            print(f"{'─'*80}")
            print(f"📘 {row['title']}")
            print(f"🏷️  Category: {row['category']}")
            print(f"📝 {row['description'][:150]}...")
            print()
        
        print("="*80)
    
    def run(self):
        """Main execution flow"""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("\n" + "="*80)
        print("🚀 AI RECOMMENDATION ENGINE".center(80))
        print("="*80)
        
        # Load data
        print("\n⚙️  Initializing recommendation system...")
        self.load_sample_data()
        print("✓ System ready with 20 learning paths")
        
        while True:
            # Get user preferences
            if not self.get_user_preferences():
                retry = input("\nTry again? (y/n): ").lower()
                if retry != 'y':
                    break
                continue
            
            # Generate recommendations
            recommendations = self.get_recommendations(top_n=5)
            
            # Display results
            self.display_recommendations(recommendations)
            
            # Continue or exit
            print("\n" + "="*80)
            choice = input("\nGet new recommendations? (y/n): ").lower()
            if choice != 'y':
                break
            
            os.system('clear' if os.name == 'posix' else 'cls')
        
        print("\n" + "="*80)
        print("Thank you for using AI Recommendation Engine!".center(80))
        print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    engine = RecommendationEngine()
    engine.run()