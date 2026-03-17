import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

class MatrixSpamDetector:
    """
    Spam detection using matrix transformations as learned in today's lecture
    M = Matrix (Transform matrix) - transforms email vectors from state to state
    """
    
    def __init__(self):
        # Initialize transformation matrix M (as per lecture: "So find M")
        self.M = None  
        self.spam_keywords = [
            'free', 'win', 'prize', 'congratulation', 'money',
            'urgent', 'limited', 'offer', 'cash', 'bonus',
            'winner', 'claim', 'million', 'dollar', 'guarantee'
        ]

    def _email_to_feature(self, email):
        """
        Convert one email into a 2D feature vector [spam_keyword_count, caps_ratio].
        """
        email_lower = email.lower()

        # Count spam keywords
        x = sum(1 for keyword in self.spam_keywords if keyword in email_lower)

        # Calculate all-caps ratio
        caps_count = sum(1 for c in email if c.isupper())
        total_chars = len(email)
        y = caps_count / max(total_chars, 1)

        return [x, y]
        
    def create_email_matrix(self, emails):
        """
        Create matrix M where columns are email vectors
        M = [v1, v2, v3, ..., vn] as shown in lecture
        """
        # Convert emails to feature vectors
        features = [self._email_to_feature(email) for email in emails]
        
        # Create matrix M where columns are vectors
        # M = [[x1, x2, x3, ..., xn],
        #      [y1, y2, y3, ..., yn]]
        return np.array(features).T
    
    def transform_coordinates(self, email_vector):
        """
        Transform email from local to global coordinate system
        As mentioned in lecture: "Both local and global coordinates are required"
        """
        # Local coordinates (raw features)
        local_coords = email_vector
        
        # Transform to global coordinates using matrix multiplication
        # VR = M * VS  (as shown in lecture: VR = M * VS)
        if self.M is not None and len(self.M) > 0:
            # Use transformation matrix to map to spam space
            global_coords = np.dot(self.M[:, :2].T, local_coords) if self.M.shape[1] >= 2 else local_coords
            return global_coords
        return local_coords
    
    def calculate_eigenvector_spam_score(self, email_vector):
        """
        Apply eigenvector concept from lecture:
        "Eigenvector dimensions will change but direction rotate coordinates but not direction will not change"
        """
        # Calculate angle of email vector
        angle = np.arctan2(email_vector[1], email_vector[0]) if email_vector[0] != 0 else 0
        
        # The eigenvector direction that doesn't change indicates spamminess
        # If vector points towards spam region, score is high
        spam_direction = np.pi / 4  # 45 degrees (spam region)
        
        # Calculate how aligned the email vector is with spam direction
        alignment = 1 - abs(angle - spam_direction) / np.pi
        
        return alignment
    
    def cartesian_to_polar_transform(self, x, y):
        """
        Transform from Cartesian to Polar coordinate system
        As per lecture: "Polar coordinate system - Rotate at any point"
        """
        r = np.sqrt(x**2 + y**2)  # distance (magnitude of spamminess)
        theta = np.arctan2(y, x)   # angle (type of spam)
        
        return r, theta
    
    def detect_spam(self, email, threshold=0.6):
        """
        Main function to detect if an email is spam
        Uses matrix transformations from today's lecture
        """
        # Step 1: Create email vector
        email_vector = self.create_email_matrix([email])[:, 0]
        x, y = email_vector
        
        print(f"\n Email Analysis:")
        print(f"   Email Vector: [{x:.2f}, {y:.2f}]")
        print(f"   x = spam keywords count: {int(x)}")
        print(f"   y = capitalization ratio: {y:.2f}")
        
        # Step 2: Transform to polar coordinates
        r, theta = self.cartesian_to_polar_transform(x, y)
        print(f"\n Polar Transformation:")
        print(f"   r (spam magnitude): {r:.2f}")
        print(f"   θ (spam angle): {theta:.2f} rad ({np.degrees(theta):.1f}°)")
        
        # Step 3: Apply eigenvector analysis
        eigen_score = self.calculate_eigenvector_spam_score(email_vector)
        print(f"\n Eigenvector Analysis:")
        print(f"   Alignment with spam direction: {eigen_score:.2f}")
        
        # Step 4: Global coordinate transformation
        global_coords = self.transform_coordinates(email_vector)
        print(f"\n Global Coordinate Transformation:")
        print(f"   Global coordinates: {global_coords}")
        
        # Step 5: Calculate final spam score using matrix operations
        # M = [v1, v2, v3, ..., vn] as per lecture
        spam_score = r * eigen_score
        
        # Apply transformation matrix if available
        if self.M is not None and self.M.shape[1] >= 2:
            # Use first two columns of transformation matrix
            M_transform = self.M[:, :2]
            transformed_score = np.dot(M_transform.T, email_vector)
            spam_score = np.mean(np.abs(transformed_score))
        
        # Normalize score to 0-1 range
        spam_score = min(max(spam_score / 10, 0), 1)
        
        print(f"\n Final Spam Score: {spam_score:.2f}")
        print(f"   Threshold: {threshold}")
        
        # Decision
        is_spam = spam_score > threshold
        
        if is_spam:
            print("\n VERDICT: SPAM EMAIL DETECTED!")
            print("   This email has been transformed into the spam region.")
        else:
            print("\n VERDICT: LEGITIMATE EMAIL")
            print("   This email remains in the safe region.")
        
        return {
            'is_spam': is_spam,
            'spam_score': spam_score,
            'email_vector': email_vector.tolist(),
            'polar_coordinates': {'r': r, 'theta': theta},
            'eigenvector_alignment': eigen_score,
            'global_coordinates': global_coords.tolist()
        }
    
    def find_transformation_matrix(self, old_emails, new_emails, old_labels, new_labels):
        """
        Find M (transformation matrix) as per lecture:
        "For new position find M first. Old and new position are known then M will automatically find"
        """
        # Create matrices for old and new states
        old_matrix = self.create_email_matrix(old_emails)
        new_matrix = self.create_email_matrix(new_emails)

        old_aligned = old_matrix
        new_aligned = new_matrix

        # If sample counts differ, align by label-wise centroids when possible.
        if old_matrix.shape[1] != new_matrix.shape[1]:
            if len(old_labels) == old_matrix.shape[1] and len(new_labels) == new_matrix.shape[1]:
                shared_labels = sorted(set(old_labels).intersection(set(new_labels)))
                if shared_labels:
                    old_centroids = []
                    new_centroids = []
                    old_labels_arr = np.array(old_labels)
                    new_labels_arr = np.array(new_labels)

                    for label in shared_labels:
                        old_class = old_matrix[:, old_labels_arr == label]
                        new_class = new_matrix[:, new_labels_arr == label]

                        if old_class.size == 0 or new_class.size == 0:
                            continue

                        old_centroids.append(np.mean(old_class, axis=1))
                        new_centroids.append(np.mean(new_class, axis=1))

                    if old_centroids and new_centroids:
                        old_aligned = np.column_stack(old_centroids)
                        new_aligned = np.column_stack(new_centroids)
                    else:
                        min_cols = min(old_matrix.shape[1], new_matrix.shape[1])
                        old_aligned = old_matrix[:, :min_cols]
                        new_aligned = new_matrix[:, :min_cols]
                else:
                    min_cols = min(old_matrix.shape[1], new_matrix.shape[1])
                    old_aligned = old_matrix[:, :min_cols]
                    new_aligned = new_matrix[:, :min_cols]
            else:
                min_cols = min(old_matrix.shape[1], new_matrix.shape[1])
                old_aligned = old_matrix[:, :min_cols]
                new_aligned = new_matrix[:, :min_cols]

        # Calculate transformation matrix M such that new = M * old
        # Using pseudoinverse to solve for M
        self.M = np.dot(new_aligned, np.linalg.pinv(old_aligned))
        
        print("\n Transformation Matrix M found:")
        print(self.M)
        
        return self.M

# Example usage with emails
def demonstrate_spam_detection():
    detector = MatrixSpamDetector()
    
    # Test emails (v1, v2, v3 as per lecture notation)
    emails = [
        "Hey, let's meet for lunch tomorrow",  # v1 - legitimate
        "CONGRATULATIONS! You've WON $1,000,000!!! CLAIM NOW!!!",  # v2 - spam
        "Meeting agenda for tomorrow's team sync",  # v3 - legitimate
        "FREE OFFER!!! Limited time!!! Act now!!!",  # v4 - spam
        "Can you review this document?",  # v5 - legitimate
        "URGENT: Your account has been compromised!!!",  # v6 - spam
    ]
    
    print("="*60)
    print("MATRIX-BASED SPAM DETECTION SYSTEM")
    print("Based on: Matrix Transformations")
    print("="*60)
    
    # Create the email matrix M = [v1, v2, v3, v4, v5, v6]
    M = detector.create_email_matrix(emails)
    print(f"\n Email Matrix M = [v1, v2, v3, v4, v5, v6]:")
    print(M)
    print("\n   Each column is an email vector:")
    print("   [spam_keywords, caps_ratio]")
    
    # Analyze each email
    for i, email in enumerate(emails, 1):
        print(f"\n{'─'*50}")
        print(f"Email v{i}:")
        print(f"Content: {email}")
        
        result = detector.detect_spam(email)
        
        print(f"{'─'*50}")
    
    # Demonstrate transformation matrix
    print("\n\n FINDING TRANSFORMATION MATRIX M")
    print("As per lecture: 'Old and new position are known then M will automatically find'")
    
    # Split emails into training and testing
    train_emails = emails[:4]
    train_labels = [0, 1, 0, 1]  # 0=ham, 1=spam
    test_emails = emails[4:]
    test_labels = [0, 1]
    
    # Find transformation matrix
    detector.find_transformation_matrix(
        train_emails, test_emails,
        train_labels, test_labels
    )

# Run the demonstration
if __name__ == "__main__":
    demonstrate_spam_detection()
    
    # Additional example with coordinate systems
    print("\n\n" + "="*60)
    print("COORDINATE SYSTEM TRANSFORMATION EXAMPLE")
    print("="*60)
    
    detector = MatrixSpamDetector()
    
    # Local coordinate system example
    print("\n Local Coordinate System (attached with the body):")
    print("   Features specific to this email only")
    
    email = "FREE MONEY!!! CLAIM YOUR PRIZE NOW!!!"
    print(f"\nEmail: {email}")
    
    # Transform through different coordinate systems
    vector = detector.create_email_matrix([email])[:, 0]
    print(f"\n1. Cartesian Coordinates (x, y): {vector}")
    
    r, theta = detector.cartesian_to_polar_transform(vector[0], vector[1])
    print(f"2. Polar Coordinates (r, θ): r={r:.2f}, θ={np.degrees(theta):.1f}°")
    
    global_coords = detector.transform_coordinates(vector)
    print(f"3. Global Coordinates (transformed): {global_coords}")
    
    eigen_score = detector.calculate_eigenvector_spam_score(vector)
    print(f"4. Eigenvector Analysis: {eigen_score:.2f}")
    
    # Final verdict
    result = detector.detect_spam(email)
    print(f"\n Final Verdict: {'SPAM' if result['is_spam'] else 'HAM'}")
