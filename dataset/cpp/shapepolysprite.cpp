#include <eepp/physics/shapepolysprite.hpp>

#ifdef PHYSICS_RENDERER_ENABLED

#include <eepp/graphics/sprite.hpp>

namespace EE { namespace Physics {

ShapePolySprite* ShapePolySprite::New( Physics::Body* body, int numVerts, cVect* verts,
									   cVect offset, Sprite* Sprite, bool AutoDeleteSprite ) {
	return eeNew( ShapePolySprite, ( body, numVerts, verts, offset, Sprite, AutoDeleteSprite ) );
}

ShapePolySprite* ShapePolySprite::New( Physics::Body* body, cpFloat width, cpFloat height,
									   Sprite* Sprite, bool AutoDeleteSprite ) {
	return eeNew( ShapePolySprite, ( body, width, height, Sprite, AutoDeleteSprite ) );
}

ShapePolySprite::ShapePolySprite( Physics::Body* body, int numVerts, cVect* verts, cVect offset,
								  Sprite* Sprite, bool AutoDeleteSprite ) :
	ShapePoly( body, numVerts, verts, offset ),
	mSprite( Sprite ),
	mSpriteAutoDelete( AutoDeleteSprite ) {
	offsetSet( centroid( numVerts, verts ) );
}

ShapePolySprite::ShapePolySprite( Physics::Body* body, cpFloat width, cpFloat height,
								  Sprite* Sprite, bool AutoDeleteSprite ) :
	ShapePoly( body, width, height ), mSprite( Sprite ), mSpriteAutoDelete( AutoDeleteSprite ) {
	mSprite->setSize( Sizef( width, height ) );
	offsetSet( cVectNew( width / 2, height / 2 ) );
}

ShapePolySprite::~ShapePolySprite() {
	if ( mSpriteAutoDelete )
		eeSAFE_DELETE( mSprite );
}

void ShapePolySprite::draw( Space* space ) {
	cVect Pos = getBody()->getPos();

	mSprite->setOffset( mOffset );
	mSprite->setPosition( Vector2f( Pos.x, Pos.y ) );
	mSprite->setRotation( getBody()->getAngleDeg() );
	mSprite->draw();
}

void ShapePolySprite::offsetSet( cVect center ) {
	cVect myCenter = cVectNew( ( mSprite->getSize().x / 2 ), ( mSprite->getSize().y / 2 ) );

	mOffset = Vector2i( ( Int32 )( -myCenter.x + ( center.x - myCenter.x ) ),
						( Int32 )( -myCenter.y + ( center.y - myCenter.y ) ) );
}

Sprite* ShapePolySprite::getSprite() const {
	return mSprite;
}

}} // namespace EE::Physics

#endif
