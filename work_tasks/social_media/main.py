from social_media_script import SocialNetworkApp
from dotenv import load_dotenv
import os

load_dotenv()
# Connect to Neo4

def main():
    app = SocialNetworkApp("bolt://localhost:7687", os.getenv('NEO4J_USERNAME'), os.getenv("NEO4J_PASSWORD"))

    while True:
        print("\n1. Register User")
        print("2. Update User")
        print("3. Add Friend")
        print("4. Unfriend")
        print("5. Send Friend Request")
        print("6. Accept Friend Request")
        print("7. Reject Friend Request")
        print("8. Create Post")
        print("9. Like Post")
        print("10. Comment on Post")
        print("11. Join Group")
        print("12. Friend Recommendations")
        print("13. Search User")
        print("14. List Friends")
        print("15. List Posts")
        print("16. List Comments")
        print("17. List Groups")
        print("18. List Group Members")
        print("19. List Friend Requests")
        print("20. Exit")

        choice = input("Choose an option: ")

        if choice == "1":
            name = input("Enter name: ")
            age = int(input("Enter age: "))
            location = input("Enter location: ")
            interests = input("Enter interests (comma separated): ")
            interests_list = interests.split(',')
            result = app.register_user(name, age, location, interests_list)
            print(f"User {name} registered.")

        elif choice == "2":
            name = input("Enter name: ")
            age = input("Enter age (leave blank to skip): ")
            location = input("Enter location (leave blank to skip): ")
            interests = input("Enter interests (comma separated, leave blank to skip): ")

            age = int(age) if age else None
            interests_list = interests.split(',') if interests else None
            result = app.update_user(name, age=age, location=location, interests=interests_list)
            print(f"User {name} updated.")

        elif choice == "3":
            user1_name = input("Enter your name: ")
            user2_name = input("Enter friend's name: ")
            result = app.add_friend(user1_name, user2_name)
            print(f"{user1_name} and {user2_name} are now friends.")

        elif choice == "4":
            user1_name = input("Enter your name: ")
            user2_name = input("Enter friend's name: ")
            result = app.unfriend(user1_name, user2_name)
            print(f"{user1_name} and {user2_name} are no longer friends.")

        elif choice == "5":
            sender_name = input("Enter your name: ")
            receiver_name = input("Enter receiver's name: ")
            result = app.send_friend_request(sender_name, receiver_name)
            print(f"Friend request sent from {sender_name} to {receiver_name}.")

        elif choice == "6":
            sender_name = input("Enter sender's name: ")
            receiver_name = input("Enter your name: ")
            result = app.accept_friend_request(sender_name, receiver_name)
            print(f"Friend request from {sender_name} accepted.")

        elif choice == "7":
            sender_name = input("Enter sender's name: ")
            receiver_name = input("Enter your name: ")
            result = app.reject_friend_request(sender_name, receiver_name)
            print(f"Friend request from {sender_name} rejected.")

        elif choice == "8":
            user_name = input("Enter your name: ")
            content = input("Enter post content: ")
            created_at = input("Enter the date (YYYY-MM-DD): ")
            result = app.create_post(user_name, content, created_at)
            print("Post created.")

        elif choice == "9":
            user_name = input("Enter your name: ")
            post_id = input("Enter post ID: ")
            result = app.like_post(user_name, post_id)
            print(f"Post {post_id} liked by {user_name}.")

        elif choice == "10":
            user_name = input("Enter your name: ")
            post_id = input("Enter post ID: ")
            comment_text = input("Enter your comment: ")
            result = app.comment_post(user_name, post_id, comment_text)
            print(f"Comment added to post {post_id}.")

        elif choice == "11":
            user_name = input("Enter your name: ")
            group_name = input("Enter group name: ")
            result = app.join_group(user_name, group_name)
            print(f"{user_name} joined the group {group_name}.")

        elif choice == "12":
            user_name = input("Enter your name: ")
            result = app.friend_recommendation(user_name)
            if result:
                print(f"Friend recommendations for {user_name}:")
                for friend in result:
                    print(friend)
            else:
                print(f"No friend recommendations for {user_name}.")

        elif choice == "13":
            name = input("Search by name (optional): ")
            location = input("Search by location (optional): ")
            interest = input("Search by interest (optional): ")
            result = app.search_user(name, location, interest)
            if result:
                print("Search results:")
                for user in result:
                    print(user)
            else:
                print("No users found.")

        elif choice == "14":
            user_name = input("Enter your name: ")
            result = app.list_friends(user_name)
            if result:
                print(f"Friends of {user_name}:")
                for friend in result:
                    print(friend)
            else:
                print(f"{user_name} has no friends listed.")

        elif choice == "15":
            result = app.list_posts()
            if result:
                print("Posts:")
                for post in result:
                    print(post)
            else:
                print("No posts available.")

        elif choice == "16":
            post_id = input("Enter post ID: ")
            result = app.list_comments(post_id)
            if result:
                print(f"Comments on post {post_id}:")
                for user, comment in result:
                    print(f"{user} commented: {comment}")
            else:
                print(f"No comments on post {post_id}.")

        elif choice == "17":
            result = app.list_groups()
            if result:
                print("Groups:")
                for group in result:
                    print(group)
            else:
                print("No groups available.")

        elif choice == "18":
            group_name = input("Enter group name: ")
            result = app.list_group_members(group_name)
            if result:
                print(f"Members of group {group_name}:")
                for member in result:
                    print(member)
            else:
                print(f"No members in group {group_name}.")

        elif choice == "19":
            user_name = input("Enter your name: ")
            result = app.list_friend_requests(user_name)
            if result:
                print(f"Friend requests for {user_name}:")
                for sender in result:
                    print(sender)
            else:
                print(f"No friend requests for {user_name}.")

        elif choice == "20":
            print("Exiting...")
            app.close()
            break

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
