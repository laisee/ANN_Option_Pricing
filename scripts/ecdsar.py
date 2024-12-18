from ecdsa import VerifyingKey, SigningKey
from ecdsa.util import sigencode_der
import ecdsa

# Generate private key and public key
private_key = SigningKey.generate(curve=ecdsa.SECP256k1)
public_key = private_key.get_verifying_key()

# Create BDN signer object
signer = ecdsa.BDNSigner(private_key)

# Sign message
signature = signer.sign_digest(b"Hello, World!", sigencode=sigencode_der)

# Verify signature
verifier = ecdsa.BDNVerifier(public_key)
if verifier.verify_digest(signature, b"Hello, World!", sigencode=sigencode_der):
    print("Signature verified!")
else:
    print("Signature verification failed!")
