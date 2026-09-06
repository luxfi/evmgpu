// Copyright (C) 2019-2025, Lux Industries, Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package warp

import (
	"github.com/luxfi/crypto/bls"
	"github.com/luxfi/ids"
	"github.com/luxfi/warp"
)

// LocalSigner implements signing with luxfi/crypto/bls
type LocalSigner struct {
	sk *bls.SecretKey
	pk *bls.PublicKey
}

// NewLocalSigner creates a new local signer using luxfi/crypto/bls
func NewLocalSigner(sk *bls.SecretKey) *LocalSigner {
	return &LocalSigner{
		sk: sk,
		pk: bls.PublicFromSecretKey(sk),
	}
}

// Sign signs the message with the private key
func (s *LocalSigner) Sign(msg []byte) ([]byte, error) {
	sig := bls.Sign(s.sk, msg)
	return bls.SignatureToBytes(sig), nil
}

// SignMessage signs a warp message over the Beam domain,
// warp.BeamSigningBytes(D), where D is the message ID. This is the domain
// warp.Signer signs and warp.VerifyEnvelope checks; signing the bare
// canonical bytes would produce a Beam no receiver accepts.
func (s *LocalSigner) SignMessage(msg *warp.Message) ([]byte, error) {
	return s.Sign(warp.BeamSigningBytes(msg.ID()))
}

// GetPublicKey returns the public key
func (s *LocalSigner) GetPublicKey() *bls.PublicKey {
	return s.pk
}

// PublicKey returns the public key as compressed bytes (48 bytes)
func (s *LocalSigner) PublicKey() []byte {
	return bls.PublicKeyToCompressedBytes(s.pk)
}

// NodeID returns the node ID derived from the public key
func (s *LocalSigner) NodeID() ids.NodeID {
	// Create a dummy NodeID for testing
	return ids.GenerateTestNodeID()
}
