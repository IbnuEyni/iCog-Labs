import streamlit as st
from movie_recommendation import recomend_by_similar_ratings

def main():
    st.title("Movie Recommendation System")

    st.header("Get Movie Recommendations Based on Similar Ratings")
    
    # User inputs
    user_id = st.number_input("Enter User ID", min_value=1)
    threshold = st.number_input("Enter Similarity Threshold", min_value=0.0, max_value=1.0, step=0.01)
    rec_number = st.number_input("Number of Recommendations", min_value=1, max_value=50, value=10)

    if st.button("Get Recommendations"):
        if user_id and threshold:
            with st.spinner("Fetching recommendations..."):
                recommendations = recomend_by_similar_ratings(user_id, threshold, rec_number)
                
                if recommendations:
                    st.success("Recommendations found!")
                    for movie in recommendations:
                        st.write(f"**Title:** {movie['title']}, **Score:** {movie['score']:.2f}")
                else:
                    st.warning("No recommendations found. Try adjusting the parameters.")
        else:
            st.error("Please provide valid inputs.")

if __name__ == "__main__":
    main()
