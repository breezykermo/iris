# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg
# Import the InstaGENI library.
import geni.rspec.igext as ig
# Import the Emulab specific extensions.
import geni.rspec.emulab as emulab

# Create a portal object,
pc = portal.Context()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

# Create the container node.
node = ig.DockerContainer("node")

node.docker_dockerfile = "https://github.com/breezykermo/iris/blob/main/docker/Dockerfile"
bs = node.Blockstore("bs", "/deep1b")
bs.size = "30GB"

# Add the node to our resource request.
request.addResource(node)

# Dump the request as an XML RSpec.
pc.printRequestRSpec(request)
