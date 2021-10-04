from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from http import HTTPStatus
import os
import pathlib
import nimblephysics as nimble
import random
import typing
import threading
from typing import List
import torch
import numpy as np


file_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'web_gui')


def createRequestHandler():
  """
  This creates a request handler that can serve the raw web GUI files, in
  addition to a configuration string of JSON.
  """
  class LocalHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, directory=file_path, **kwargs)

    def do_GET(self):
      """
      if self.path == '/json':
          resp = jsonConfig.encode("utf-8")
          self.send_response(HTTPStatus.OK)
          self.send_header("Content-type", "application/json")
          self.send_header("Content-Length", len(resp))
          self.end_headers()
          self.wfile.write(resp)
      else:
          super().do_GET()
      """
      super().do_GET()
  return LocalHTTPRequestHandler


class NimbleGUI:
  def __init__(self, worldToCopy: nimble.simulation.World):
    self.world = worldToCopy.clone()
    self.guiServer = nimble.server.GUIWebsocketServer()
    self.guiServer.renderWorld(self.world)
    # Set up the realtime animation
    self.ticker = nimble.realtime.Ticker(self.world.getTimeStep() * 1.0)
    self.ticker.registerTickListener(self._onTick)
    self.guiServer.registerConnectionListener(self._onConnect)

    self.looping = False
    self.posMatrixToLoop = np.zeros((self.world.getNumDofs(), 0))
    self.crs = []
    self.i = 0

  def serve(self, port):
    self.guiServer.serve(8070)
    server_address = ('', port)
    self.httpd = ThreadingHTTPServer(server_address, createRequestHandler())
    print('Web GUI serving on http://localhost:'+str(port))
    t = threading.Thread(None, self.httpd.serve_forever)
    t.daemon = True
    t.start()

  def stopServing(self):
    self.guiServer.stopServing()
    self.httpd.shutdown()

  def displayState(self, state: torch.Tensor):
    self.looping = False
    self.world.setState(state.detach().numpy())
    self.guiServer.renderWorld(self.world)

  def loopStates(self, states: List[torch.Tensor]):
    self.looping = True
    self.statesToLoop = states
    dofs = self.world.getNumDofs()
    poses = np.zeros((dofs, len(states)))
    for i in range(len(states)):
      # Take the top-half of each state vector, since this is the position component
      poses[:, i] = states[i].detach().numpy()[:dofs]
    self.guiServer.renderTrajectoryLines(self.world, poses)
    self.posMatrixToLoop = poses

  def loopStatesWithContacts(self, states: List[torch.Tensor], crs: List[nimble.collision.CollisionResult]):
    self.looping = True
    self.statesToLoop = states
    dofs = self.world.getNumDofs()
    poses = np.zeros((dofs, len(states)))
    for i in range(len(states)):
      # Take the top-half of each state vector, since this is the position component
      poses[:, i] = states[i].detach().numpy()[:dofs]
    # self.guiServer.renderTrajectoryLines(self.world, poses)
    self.posMatrixToLoop = poses

    if len(crs) > 0:
      assert len(states) == len(crs)
      self.crs = crs

    self.nativeAPI().setAutoflush(False)    # otherwise slow with all the contact lines

  def loopPosMatrix(self, poses: np.ndarray):
    self.looping = True
    self.guiServer.renderTrajectoryLines(self.world, poses)
    # It's important to make a copy, because otherwise we get a reference to internal C++ memory that gets cleared
    self.posMatrixToLoop = np.copy(poses)

  def stopLooping(self):
    self.looping = False

  def nativeAPI(self) -> nimble.server.GUIWebsocketServer:
    return self.guiServer

  def blockWhileServing(self):
    self.guiServer.blockWhileServing()

  def _onTick(self, now):
    if self.looping:
      if self.i < np.shape(self.posMatrixToLoop)[1]:

        self.world.setPositions(self.posMatrixToLoop[:, self.i])
        self.guiServer.renderWorld(self.world)

        if len(self.crs) > 0:
          cr: nimble.collision.CollisionResult = self.crs[self.i]

          for n_c in range(cr.getNumContacts()):
            cf = cr.getContact(n_c).force
            cl = cr.getContact(n_c).point
            cf_end = [cl[0] + cf[0], cl[1] + cf[1], cl[2] + cf[2]]
            # cl = [0,0,0]
            # cf_end = [0,0,3]
            self.nativeAPI().createLine("a"+str(n_c), [cl, cf_end], [1, 0, 0])

        self.nativeAPI().flush()

        if len(self.crs) > 0:
          for n_c in range(cr.getNumContacts()):
            self.nativeAPI().deleteObject("a"+str(n_c))

        self.i += 1
      else:
        self.i = 0

  def _onConnect(self):
    self.ticker.start()
