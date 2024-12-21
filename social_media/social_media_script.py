from neo4j import GraphDatabase

class SocialNetworkApp:
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # Helper method for error handling
    def _handle_error(self, result):
        if not result:
            raise Exception("Operation failed or returned no results.")
        return result

    # User Registration
    def register_user(self, name, age, location, interests_list):
        if not name or not isinstance(age, int) or not location:
            raise ValueError("Invalid user input")
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    CREATE (u:User {name: $name, age: $age, location: $location, interests: $interests})
                    RETURN u
                """, name=name, age=age, location=location, interests=interests_list)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error registering user: {e}")

    # Update User Info
    def update_user(self, name, age=None, location=None, interests=None):
        updates = []
        params = {"name": name}
        
        if age:
            updates.append("u.age = $age")
            params["age"] = age
        if location:
            updates.append("u.location = $location")
            params["location"] = location
        if interests:
            updates.append("u.interests = $interests")
            params["interests"] = interests
        
        if not updates:
            raise ValueError("No update fields provided")

        update_cypher = ", ".join(updates)
        try:
            with self.driver.session() as session:
                result = session.run(f"""
                    MATCH (u:User {{name: $name}})
                    SET {update_cypher}
                    RETURN u
                """, **params)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error updating user: {e}")

    # Add Friend
    def add_friend(self, user1_name, user2_name):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u1:User {name: $user1_name}), (u2:User {name: $user2_name})
                    CREATE (u1)-[:FRIENDS_WITH]->(u2)
                    RETURN u1, u2
                """, user1_name=user1_name, user2_name=user2_name)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error adding friend: {e}")

    # Unfriend
    def unfriend(self, user1_name, user2_name):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u1:User {name: $user1_name})-[r:FRIENDS_WITH]-(u2:User {name: $user2_name})
                    DELETE r
                    RETURN u1, u2
                """, user1_name=user1_name, user2_name=user2_name)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error unfriending: {e}")

    # Send Friend Request
    def send_friend_request(self, sender_name, receiver_name):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (sender:User {name: $sender_name}), (receiver:User {name: $receiver_name})
                    CREATE (sender)-[:OUTGOING_REQUEST]->(receiver), (receiver)-[:INCOMING_REQUEST]->(sender)
                    RETURN sender, receiver
                """, sender_name=sender_name, receiver_name=receiver_name)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error sending friend request: {e}")

    # Accept Friend Request
    def accept_friend_request(self, sender_name, receiver_name):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (sender:User {name: $sender_name})-[req:INCOMING_REQUEST]->(receiver:User {name: $receiver_name})
                    DELETE req 
                    CREATE (sender)-[:FRIENDS_WITH]->(receiver)
                    RETURN receiver, sender
                """, sender_name=sender_name, receiver_name=receiver_name)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error accepting friend request: {e}")

    # Reject Friend Request
    def reject_friend_request(self, sender_name, receiver_name):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (sender:User {name: $sender_name})-[req:INCOMING_REQUEST]->(receiver:User {name: $receiver_name})
                    DELETE req 
                    RETURN receiver, sender
                """, sender_name=sender_name, receiver_name=receiver_name)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error rejecting friend request: {e}")

    # Create Post
    def create_post(self, user_name, post_id, content, created_at):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User {name: $user_name})
                    CREATE (p:Post {post_id: $post_id, content: $content, created_at: $created_at})<-[:POSTED]-(u)
                    RETURN p
                """, user_name=user_name, post_id=post_id, content=content, created_at=created_at)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error creating post: {e}")

    # Like Post
    def like_post(self, user_name, post_id):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User {name: $user_name}), (p:Post {post_id: $post_id})
                    CREATE (u)-[:LIKES]->(p)
                    RETURN u, p
                """, user_name=user_name, post_id=post_id)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error liking post: {e}")

    # Comment on Post
    def comment_post(self, user_name, post_id, comment_text):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User {name: $user_name}), (p:Post {post_id: $post_id})
                    CREATE (u)-[:COMMENTS {text: $comment_text}]->(p)
                    RETURN u, p
                """, user_name=user_name, post_id=post_id, comment_text=comment_text)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error commenting on post: {e}")

    # Create Group
    def create_group(self, group_name, description):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    CREATE (g:Group {name: $group_name, description: $description})
                    RETURN g
                """, group_name=group_name, description=description)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error creating group: {e}")

    # Join Group
    def join_group(self, user_name, group_name):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User {name: $user_name}), (g:Group {name: $group_name})
                    CREATE (u)-[:JOINED]->(g)
                    RETURN u, g
                """, user_name=user_name, group_name=group_name)
                return self._handle_error(result.single())
        except Exception as e:
            print(f"Error joining group: {e}")

    # Friend Recommendation
    def friend_recommendation(self, user_name):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User {name: $user_name})-[:FRIENDS_WITH]-(friend)-[:FRIENDS_WITH]-(recommended)
                    WHERE NOT (u)-[:FRIENDS_WITH]-(recommended) AND u <> recommended
                    RETURN recommended, COUNT(friend) AS mutual_friends
                    ORDER BY mutual_friends DESC
                    LIMIT 5
                """, user_name=user_name)
                return self._handle_error(result.value())
        except Exception as e:
            print(f"Error with friend recommendations: {e}")

    # Search User
    def search_user(self, name=None, location=None, interest=None):
        params = {}
        filters = []
        
        if name:
            filters.append("u.name CONTAINS $name")
            params["name"] = name
        if location:
            filters.append("u.location CONTAINS $location")
            params["location"] = location
        if interest:
            filters.append("ANY(i IN u.interests WHERE i CONTAINS $interest)")
            params["interest"] = interest
            
        filter_query = " AND ".join(filters) if filters else "TRUE"
        
        try:
            with self.driver.session() as session:
                result = session.run(f"""
                    MATCH (u:User)
                    WHERE {filter_query}
                    RETURN u
                """, **params)
                return self._handle_error(result.value())
        except Exception as e:
            print(f"Error searching user: {e}")


    def list_likes(self, post_id):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[:LIKES]->(p:Post {id: $post_id})
                RETURN u
            """, post_id=post_id)
            return result.values("u")

    def list_comments(self, post_id):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[c:COMMENTS]->(p:Post {id: $post_id})
                RETURN u, c.comment AS comment
            """, post_id=post_id)
            return result.values("u", "comment")
    
    def list_friends(self, user_name):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {name: $user_name})-[:FRIENDS_WITH]-(f:User)
                RETURN f
            """, user_name=user_name)
            return result.values("f")

    def list_groups(self):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (g:Group)
                RETURN g
            """)
            return result.values("g")

    def list_group_members(self, group_name):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[:JOINS]->(g:Group {name: $group_name})
                RETURN u
            """, group_name=group_name)
            return result.values("u")

    def list_friend_requests(self, user_name):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {name: $user_name})-[:INCOMING_REQUEST]-(sender:User)
                RETURN sender
            """, user_name=user_name)
            return result.values("sender")