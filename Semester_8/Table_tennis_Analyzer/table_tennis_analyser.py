import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from datetime import datetime
import time
import os

class Rally:
    def __init__(self):
        """Initialize a new rally tracking object"""
        self.bounces = []  # List of (position, side, timestamp)
        self.started = datetime.now()
        self.ended = None
        self.winner = None
        self.reason = None
        
    def add_bounce(self, position, side):
        """Add a bounce with position, side, and current timestamp"""
        self.bounces.append((position, side, datetime.now()))
        
    def end_rally(self, winner, reason):
        """End the current rally with winner and reason"""
        self.ended = datetime.now()
        self.winner = winner
        self.reason = reason
        
    def get_last_bounce_side(self):
        """Get the side of the last bounce"""
        if not self.bounces:
            return None
        return self.bounces[-1][1]
    
    def get_bounce_count_on_side(self, side):
        """Count bounces on a specific side"""
        return sum(1 for bounce in self.bounces if bounce[1] == side)
    
    def time_since_last_bounce(self):
        """Calculate time since last bounce"""
        if not self.bounces:
            return float('inf')
        return (datetime.now() - self.bounces[-1][2]).total_seconds()

class TableTennisAnalyzer:
    def __init__(self, model_path, video_path, confidence=0.4):
        """Initialize analyzer with model, video path, and confidence threshold"""
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.confidence = confidence
        
        # Object class mapping
        self.class_names = {0: 'table', 1: 'net', 2: 'ball'}
        
        # Visualization colors
        self.colors = {
            'table': (0, 255, 0),  # Green
            'net': (255, 0, 0),    # Blue
            'ball': (0, 0, 255),   # Red
            'trajectory': (255, 255, 0)  # Yellow
        }
        
        # Tracking variables
        self.ball_positions = []
        self.max_positions = 15
        self.bounce_positions = []
        self.spots = []  # For bounce spots with timestamps
        self.max_bounce_positions = 5
        
        # Speed tracking variables
        self.n = 0
        self.event1 = None
        self.event2 = None
        self.speed = 0
        self.list_speed = []
        self.avg_speed = 0
        self.frames_without_ball = 0
        
        # Game state
        self.player1_score = 0
        self.player2_score = 0
        self.table_coords = None
        self.net_coords = None
        self.timeout_threshold = 3  # seconds
        
        # Rally tracking
        self.current_rally = Rally()
        self.rally_history = []
        self.rally_in_progress = False
        self.serving_side = None  # 'left' or 'right'
        
        # State tracking
        self.state = 'WAITING_FOR_SERVE'  # States: WAITING_FOR_SERVE, RALLY_IN_PROGRESS, POINT_SCORED
        self.last_state_change = datetime.now()
        self.last_net_crossing = None

    def detect_objects(self, frame):
        """Detect table, net, and ball in frame using YOLO"""
        results = self.model(frame, conf=self.confidence)
        
        objects = {'table': None, 'net': None, 'ball': None}
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                objects[class_name] = {
                    'box': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'confidence': float(box.conf[0])
                }
        
        return objects

    def define_table_sides(self, table_box, net_box):
        """Divide table into left/right sides based on net position"""
        if not table_box or not net_box:
            return None
            
        t_x1, t_y1, t_x2, t_y2 = table_box
        n_x1, n_y1, n_x2, n_y2 = net_box
        net_center = (n_x1 + n_x2) / 2
        
        return {
            'left': {'x1': t_x1, 'y1': t_y1, 'x2': net_center, 'y2': t_y2},
            'right': {'x1': net_center, 'y1': t_y1, 'x2': t_x2, 'y2': t_y2}
        }

    def is_ball_on_table(self, ball_pos, table_sides):
        """Check if ball is on table and which side"""
        if not ball_pos or not table_sides:
            return None
            
        x, y = ball_pos
        left = table_sides['left']
        right = table_sides['right']
        
        # Add a small margin around the table
        margin = 15  # pixels
        
        if (left['x1'] - margin <= x <= left['x2'] + margin and 
            left['y1'] - margin <= y <= left['y2'] + margin):
            return 'left'
        if (right['x1'] - margin <= x <= right['x2'] + margin and 
            right['y1'] - margin <= y <= right['y2'] + margin):
            return 'right'
        return None

    def calculate_time(self):
        """Calculate time between events"""
        if self.n % 2 == 0 and self.n != 0:
            self.event1 = datetime.now()
            time_diff = (self.event1 - self.event2).total_seconds()
        elif self.n % 2 == 1:
            self.event2 = datetime.now()
            time_diff = (self.event2 - self.event1).total_seconds()
        else:
            self.event1 = datetime.now()
            time_diff = 0

        self.n += 1
        return time_diff

    def calculate_distance(self, table_coords):
        """Calculate distance in meters"""
        if len(self.ball_positions) >= 2 and table_coords:
            t_x1, t_y1, t_x2, t_y2 = table_coords
            
            # Calculate table width in pixels
            table_width_px = t_x2 - t_x1
            
            # Standard table tennis table is 2.74m wide
            px_to_m = 2.74 / table_width_px
            
            # Calculate distance between last two ball positions
            p1 = self.ball_positions[-2]
            p2 = self.ball_positions[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance = np.sqrt(dx*dx + dy*dy) * px_to_m
            return distance
        else:
            return 0

    def calculate_avg_speed(self):
        """Calculate average speed over last 5 readings"""
        if len(self.list_speed) > 5:
            self.avg_speed = sum(self.list_speed[-5:]) / 5

    def detect_bounce(self, table_sides):
        """Detect ball bounce with improved duplicate detection"""
        if len(self.ball_positions) < 5:  # Require more positions for reliable detection
            return None, None
        
        # Use more positions to get a reliable trajectory pattern
        positions = self.ball_positions[-5:]
        
        # Check if we have a clear downward then upward pattern
        # Using more points helps eliminate noise
        y_values = [p[1] for p in positions]
        
        # Look for a clear V-shaped pattern in the y-values
        # Where the ball comes down then goes back up
        is_bounce = False
        bounce_idx = 0
        
        for i in range(1, len(y_values)-1):
            if (y_values[i] > y_values[i-1] and 
                y_values[i] > y_values[i+1] and
                y_values[i] - y_values[i-1] > 5 and  # Must be significant movement
                y_values[i] - y_values[i+1] > 5):
                is_bounce = True
                bounce_idx = i
                break
        
        if is_bounce:
            bounce_pos = positions[bounce_idx]
            bounce_side = self.is_ball_on_table(bounce_pos, table_sides)
            
            # Only count if on table
            if bounce_side:
                # Use a timestamp-based approach to avoid double counting
                current_time = datetime.now()
                
                # Only register a new bounce if enough time has passed since last one
                # Or if it's far enough away from previous bounce
                min_distance = 30  # pixels
                min_time_diff = 0.5  # seconds
                
                should_register = True
                if self.spots:
                    last_bounce_pos, _, last_bounce_time = self.spots[-1]
                    time_diff = (current_time - last_bounce_time).total_seconds()
                    distance = self.euclidean_distance(bounce_pos, last_bounce_pos)
                    
                    if time_diff < min_time_diff or distance < min_distance:
                        should_register = False
                        
                if should_register:
                    self.spots.append((bounce_pos, bounce_side, current_time))
                    
                    # Keep list at reasonable size
                    if len(self.spots) > self.max_bounce_positions:
                        self.spots.pop(0)
                    
                    print(f"Bounce detected on {bounce_side} side")
                    
                    # If rally is in progress, add bounce to current rally
                    if self.state == 'RALLY_IN_PROGRESS':
                        self.current_rally.add_bounce(bounce_pos, bounce_side)
                        
                    return True, bounce_side
        
        return None, None

    def euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_net_crossing(self, prev_pos, curr_pos):
        """Check if ball crossed the net"""
        if not self.net_coords or not prev_pos or not curr_pos:
            return False
            
        net_x1, _, net_x2, _ = self.net_coords
        net_center = (net_x1 + net_x2) / 2
        
        # Ball crossed from left to right
        if prev_pos[0] < net_center and curr_pos[0] > net_center:
            return 'left_to_right'
        # Ball crossed from right to left
        elif prev_pos[0] > net_center and curr_pos[0] < net_center:
            return 'right_to_left'
        
        return None

    def is_ball_out_of_bounds(self, ball_pos):
        """Check if ball is out of play area"""
        if not self.table_coords:
            return False
            
        table_x1, table_y1, table_x2, table_y2 = self.table_coords
        margin = 100  # pixels
        
        # Check if ball is far below table or beyond sides
        return (ball_pos[1] > table_y2 + margin or
                ball_pos[0] < table_x1 - margin or 
                ball_pos[0] > table_x2 + margin)

    def update_game_state(self, ball_pos, table_sides, current_time):
        """Update game state with more robust point assignment"""
        # If no ball detected for a while, check if rally should end
        if not ball_pos:
            time_since_state_change = (current_time - self.last_state_change).total_seconds()
            
            # If we're in a rally and ball disappeared, check for timeout
            if self.state == 'RALLY_IN_PROGRESS' and time_since_state_change > self.timeout_threshold:
                if self.current_rally.bounces and len(self.current_rally.bounces) >= 2:
                    last_bounce_side = self.current_rally.get_last_bounce_side()
                    
                    # Award point based on last bounce side
                    if last_bounce_side == 'right':
                        self.award_point('player1', 'ball_disappeared_after_right_bounce')
                    else:
                        self.award_point('player2', 'ball_disappeared_after_left_bounce')
            return
            
        # Check for bounces if we have a table
        if table_sides and len(self.ball_positions) >= 5:  # Need more positions for reliable detection
            bounce_detected, bounce_side = self.detect_bounce(table_sides)
            
            # Handle rally based on state
            if self.state == 'WAITING_FOR_SERVE':
                if bounce_detected:
                    # First bounce of serve detected
                    self.state = 'RALLY_IN_PROGRESS'
                    self.serving_side = bounce_side
                    self.last_state_change = current_time
                    self.rally_in_progress = True
                    # Clear any existing spots to start fresh
                    if len(self.spots) > 1:
                        self.spots = [self.spots[-1]]
                    print(f"Rally started with serve from {bounce_side} side")
            
            elif self.state == 'RALLY_IN_PROGRESS':
                # Check double bounce rule - but only if we have at least 3 registered bounces
                # This helps eliminate false positives
                if bounce_detected and len(self.spots) >= 3:
                    # Get last two bounces
                    bounce1 = self.spots[-2]
                    bounce2 = self.spots[-1]
                    
                    # If both are on the same side
                    if bounce1[1] == bounce2[1]:
                        bounce_side = bounce1[1]
                        # And time between them is reasonable
                        time_diff = (bounce2[2] - bounce1[2]).total_seconds()
                        if 0.2 < time_diff < 2.0:  # Reasonable time for double bounce
                            if bounce_side == 'right':
                                self.award_point('player1', 'double_bounce_right')
                            else:
                                self.award_point('player2', 'double_bounce_left')
                
                # Check for net crossings during rally
                if len(self.ball_positions) >= 2:
                    crossing = self.is_net_crossing(self.ball_positions[-2], ball_pos)
                    if crossing:
                        self.last_net_crossing = crossing
                
                # Check timeout since last bounce
                if self.current_rally.bounces and self.current_rally.time_since_last_bounce() > self.timeout_threshold:
                    last_bounce_side = self.current_rally.get_last_bounce_side()
                    if last_bounce_side == 'right':
                        self.award_point('player1', 'timeout_after_right_bounce')
                    else:
                        self.award_point('player2', 'timeout_after_left_bounce')
                
                # Check for ball out of bounds
                if self.is_ball_out_of_bounds(ball_pos):
                    # Award point based on which side the ball went out from
                    if self.net_coords:
                        net_center = (self.net_coords[0] + self.net_coords[2]) / 2
                        if self.ball_positions[-2][0] < net_center:  # Coming from left side
                            self.award_point('player2', 'out_of_bounds_from_left')
                        else:  # Coming from right side
                            self.award_point('player1', 'out_of_bounds_from_right')

    def award_point(self, winner, reason):
        """Award a point to a player and reset for next rally"""
        if winner == 'player1':
            self.player1_score += 1
            print(f"Point to Player 1 ({reason}). Score: {self.player1_score}-{self.player2_score}")
        else:
            self.player2_score += 1
            print(f"Point to Player 2 ({reason}). Score: {self.player1_score}-{self.player2_score}")
        
        # End current rally
        self.current_rally.end_rally(winner, reason)
        self.rally_history.append(self.current_rally)
        
        # Reset for next rally
        self.current_rally = Rally()
        self.rally_in_progress = False
        self.state = 'WAITING_FOR_SERVE'
        self.last_state_change = datetime.now()
        self.serving_side = None
        
        # Clear tracking data for visual clarity, but keep some history
        if len(self.ball_positions) > 5:
            self.ball_positions = self.ball_positions[-5:]

    def draw_objects(self, frame, objects, table_sides):
        """Draw detected objects and game info on frame"""
        # Draw table
        if objects['table']:
            x1, y1, x2, y2 = objects['table']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['table'], 2)
            cv2.putText(frame, 'Table', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['table'], 2)

        # Draw net
        if objects['net']:
            x1, y1, x2, y2 = objects['net']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['net'], 2)
            cv2.putText(frame, 'Net', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['net'], 2)

        # Draw table sides
        if table_sides:
            left = table_sides['left']
            cv2.rectangle(frame, (int(left['x1']), int(left['y1'])), 
                        (int(left['x2']), int(left['y2'])), (50, 150, 50), 1)
            right = table_sides['right']
            cv2.rectangle(frame, (int(right['x1']), int(right['y1'])), 
                        (int(right['x2']), int(right['y2'])), (50, 50, 150), 1)

        # Draw ball and trajectory
        if objects['ball']:
            x1, y1, x2, y2 = objects['ball']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['ball'], 2)
            cv2.putText(frame, 'Ball', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['ball'], 2)
            
            # Draw trajectory
            if len(self.ball_positions) > 1:
                for i in range(1, len(self.ball_positions)):
                    cv2.line(frame, 
                            tuple(map(int, self.ball_positions[i-1])),
                            tuple(map(int, self.ball_positions[i])),
                            self.colors['trajectory'], 2)

        # Draw bounce spots
        for spot_info in self.spots:
            spot = spot_info[0]  # Get just the position
            # Color based on side: red for right, blue for left
            color = (0, 0, 255) if spot_info[1] == 'right' else (255, 0, 0)
            cv2.circle(frame, tuple(map(int, spot)), 8, color, -1)  # Filled circle

        # Draw scores with consistent positioning
        score_y_pos = 75
        cv2.putText(frame, f"Player 1: {self.player1_score}", 
                   (55, score_y_pos), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 2)
        
        # Calculate safe position for player 2 score
        player2_text = f"Player 2: {self.player2_score}"
        player2_text_size = cv2.getTextSize(player2_text, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 2)[0]
        player2_x_pos = frame.shape[1] - player2_text_size[0] - 55  # safe margin from right edge
        cv2.putText(frame, player2_text, 
                   (player2_x_pos, score_y_pos), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 2)
        
        # Draw game state
        state_text = f"State: {self.state}"
        cv2.putText(frame, state_text, 
                   (55, score_y_pos + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Draw ball speed
        if self.avg_speed > 0:
            speed_text = f"Ball Speed: {self.avg_speed:.2f} m/sec"
            cv2.putText(frame, speed_text,
                       (55, score_y_pos + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def process_video(self, output_path=None):
        """Main video processing loop"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video: {self.video_path}")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Setup video writer if output path specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every other frame for performance
            if frame_count % 2 == 0:
                current_time = datetime.now()
                
                # Detect objects
                objects = self.detect_objects(frame)
                
                # Update table and net coordinates
                if objects['table']:
                    self.table_coords = objects['table']['box']
                if objects['net']:
                    self.net_coords = objects['net']['box']
                
                # Define table sides if we have both
                table_sides = None
                if self.table_coords and self.net_coords:
                    table_sides = self.define_table_sides(self.table_coords, self.net_coords)
                
                ball_position = None
                
                # Track ball if detected
                if objects['ball']:
                    self.frames_without_ball = 0
                    ball_position = objects['ball']['center']
                    
                    # Clear old positions if this is a new rally
                    if self.state == 'WAITING_FOR_SERVE' and len(self.ball_positions) > 5:
                        self.ball_positions = []
                    
                    # Add to ball positions if movement is significant
                    if len(self.ball_positions) == 0 or self.euclidean_distance(ball_position, self.ball_positions[-1]) > 5:
                        self.ball_positions.append(ball_position)
                        
                        # Calculate speed if we have enough positions
                        if len(self.ball_positions) >= 2:
                            time_diff = self.calculate_time()
                            distance = self.calculate_distance(self.table_coords)
                            
                            if time_diff > 0:
                                self.speed = distance / time_diff
                                if 0.5 < self.speed < 30:  # Filter unrealistic speeds
                                    self.list_speed.append(self.speed)
                                    self.calculate_avg_speed()
                    
                    # Maintain position buffer - keep fewer positions to avoid ghost tracks
                    if len(self.ball_positions) > 10:  # Reduced from max_positions
                        self.ball_positions = self.ball_positions[-10:]
                else:
                    self.frames_without_ball += 1
                    # Clear ball positions more aggressively
                    if self.frames_without_ball > 5:  # Reduced threshold
                        self.ball_positions = []
                        self.frames_without_ball = 0
                
                # Update game state
                self.update_game_state(ball_position, table_sides, current_time)
                
                # Draw visualization
                self.draw_objects(frame, objects, table_sides)
                
                # Write to output if specified
                if output_path:
                    out.write(frame)
                
                # Display frame
                cv2.imshow('Table Tennis Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Table Tennis Analysis')
    parser.add_argument('--model', required=True, help='Path to YOLO model')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', help='Path to output video (optional)')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    
    args = parser.parse_args()
    
    analyzer = TableTennisAnalyzer(args.model, args.video, args.conf)
    analyzer.process_video(args.output)

if __name__ == "__main__":
    main()