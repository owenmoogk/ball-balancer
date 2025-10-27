MOTOR_LINK_LEN = 10
BAR_LINK_LEN = 10

class Ball:
  accel = (0,0,0)
  vel = (0,0,0)
  pos = (0,0,0)
  radius = 10
  
  def update(self, dt, planePosition):
    pass

class Link:
  def __init__(self, contact_point):
    self.contact_point = contact_point # (x, y)
    self.l1 = MOTOR_LINK_LEN
    self.l2 = BAR_LINK_LEN

class StewartPlatformSimulator:

  def __init__(self ,planePosition, dt):
    # add contact points
    self.dt = dt
    self.planePosition = planePosition # (theta, phi, z_offset)
    self.links = [Link(), Link(), Link()]
    self.ball = Ball()

  def get_motor_angles():
    # do math (use self.links)


    return (angle1, angle2, angle3)

  def step(self, targetPlanePosition):
    # for now (add inertia or somethings)
    self.planePosition = targetPlanePosition

    self.ball.update()

    
    

