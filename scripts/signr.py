from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

# Generate private key
private_key = ec.generate_private_key(ec.SECP256K1())

# Generate public key
public_key = private_key.public_key()

# Serialize public key
public_key_bytes = public_key.public_bytes(
    encoding=serialization.Encoding.X962,
    format=serialization.PublicFormat.UncompressedPoint
)

# Sign message
signature = private_key.sign(
    b"Hello, World!",
    ec.ECDSA(hashes.SHA256())
)
print(signature)

# Verify signature
try:
    public_key.verify( signature, b"Hell0, World!", ec.ECDSA(hashes.SHA256()))
    print("Verified signature with Success")
except Exception as ex:
    print("Error while verifying siggy: ",ex)
