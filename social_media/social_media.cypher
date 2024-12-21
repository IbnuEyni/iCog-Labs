CREATE (u1:User {name: $name, age: $age, location: $location, interests: $interests})

CREATE (u2:User {name: $name, age: $age, location: $location, interests: $interests})
CREATE (g:Group {name: $name, members: $members})
CREATE (p: Post {post_id: $post_id, content: $content, created_at: $created_at})
CREATE (comment: COMMENTS {text: $user_comment})
// RETURN u2, u1

// WITH u1, u2

// friends
MATCH (u1:User {name: $user1_name}), (u2:User {name: $user2_name})
CREATE (u1)-[:FRIENDS_WITH]->(u2)
// RETURN u1.name, u2.name

//Remove 
MATCH (u1:User {name: $user1_name})-[r:FRIENDS_WITH]-(u2:User {name: $user2_name})
DELETE r
// RETURN u1, u2

//send friend request 
MATCH (sender:User {name: $sender_name}), (receiver:User {name: $receiver_name})
CREATE (sender)-[:OUTGOING_REQUEST]->(receiver), (receiver)-[:INCOMING_REQUEST]->(sender)
// RETURN sender, receiver 

//accept friend request
MATCH(sender:User {name: $sender_name})-[req:INCOMING_REQUEST]->(receiver:User {name: $receiver_name})
DELETE req
CREATE(sender)-[:FRIENDS_WITH]->(receiver)
// RETURN sender, receiver

//reject friend request
MATCH(sender:User {name: $sender_name})-[req:INCOMING_REQUEST]->(receiver:User {name: $receiver_name})
DELETE req
// RETURN receiver, sender


// create post
MATCH (u:User {name: $user_name})
CREATE (p:Post {post_id: $post_id, content: $content, created_at: $created_at})<-[:POSTED]-(u)
// RETURN p

// like post
MATCH(u:User {name: $user_name}), (p: Post {id:$post_id})
CREATE(u)-[LIKES]->(p)
// RETURN u, p

//comment on a post
MATCH(u:User {name: $user_name}), (p: Post {id:$post_id})
CREATE(u)-[COMMENTS {text: $user_comment}]->(p)
// RETURN u, p

// join group
MATCH (u:User {name: $user_name}), (g:Group {name: $group_name, members:$members})
CREATE (u)-[:JOINED]->(g)
// RETURN u, g

// friend recommendation
 MATCH(u:User {name: $user_name})-[:FRIENDS_WITH]-(friend)-[:FRIENDS_WITH]-(recommended)
 WHERE NOT (u)-[:FRIENDS_WITH]-(recommended) AND u <> recommended
 RETURN recommended, COUNT(friend) AS mutual_friends
 ORDER BY mutual_friends DESC
 LIMIT 5

//  User Search
MATCH (u:User)
WHERE u.name CONTAINS $name OR u.location CONTAINS $location OR u.interests CONTAINS $interest
// RETURN u


