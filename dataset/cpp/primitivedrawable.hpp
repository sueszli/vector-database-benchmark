#ifndef EE_GRAPHICS_PRIMITIVEDRAWABLE
#define EE_GRAPHICS_PRIMITIVEDRAWABLE

#include <eepp/graphics/drawable.hpp>
#include <eepp/graphics/primitivetype.hpp>

namespace EE { namespace Graphics {

class VertexBuffer;

class EE_API PrimitiveDrawable : public Drawable {
  public:
	virtual ~PrimitiveDrawable();

	virtual void draw( const Vector2f& position, const Sizef& size );

	/** Set the fill mode used to draw primitives */
	virtual void setFillMode( const PrimitiveFillMode& Mode );

	/** @return The fill mode used to draw primitives */
	const PrimitiveFillMode& getFillMode() const;

	/** Set the blend mode used to draw primitives */
	virtual void setBlendMode( const BlendMode& Mode );

	/** @return The blend mode used to draw primitives */
	const BlendMode& getBlendMode() const;

	/** Set the line width to draw primitives */
	virtual void setLineWidth( const Float& width );

	/** @return The line with to draw primitives */
	const Float& getLineWidth() const;

	/** @return True if polygon and line smoothing is enabled */
	bool isSmooth() const;

	/** Enables/Disables polygon and line smoothing */
	void setSmooth( bool smooth );

  protected:
	PrimitiveDrawable( Type drawableType );

	PrimitiveFillMode mFillMode;
	BlendMode mBlendMode;
	Float mLineWidth;
	bool mNeedsUpdate;
	bool mRecreateVertexBuffer;
	bool mSmooth{ false };
	VertexBuffer* mVertexBuffer;

	virtual void onAlphaChange();

	virtual void onColorFilterChange();

	virtual void onPositionChange();

	void prepareVertexBuffer( const PrimitiveType& drawableType );

	virtual void updateVertex() = 0;
};

}} // namespace EE::Graphics

#endif
