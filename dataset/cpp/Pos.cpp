#include "PrecompiledHeader.h"

#include "JRenderer.h"
#include "Pos.h"

Pos::Pos(float x, float y, float z, float t, float a) :
    actX(x), actY(y), actZ(z), actT(t), actA(a), x(x), y(y), zoom(z), t(t), alpha(a)
{
    mask = 0;
}
void Pos::Update(float dt)
{
    if (dt > 0.05f) dt = 0.05f;
    actX += 10 * dt * (x - actX);
    actY += 10 * dt * (y - actY);
    actT += 10 * dt * (t - actT);
    actZ += 10 * dt * (zoom - actZ);
    actA += 10 * dt * (alpha - actA);
}

void Pos::UpdateNow()
{
    actX = x;
    actY = y;
    actT = t;
    actZ = zoom;
    actA = alpha;
}
void Pos::Render()
{
}
void Pos::Render(JQuad* quad)
{
    quad->SetColor(ARGB((int)actA, 255, 255, 255));
    JRenderer::GetInstance()->RenderQuad(quad, actX, actY, actT, actZ, actZ);
    if (mask && !actT)
        JRenderer::GetInstance()->FillRect(actX,actY,actZ * quad->mWidth, actZ* quad->mHeight, mask);
    width = quad->mWidth;
    height = quad->mHeight;
        
}
