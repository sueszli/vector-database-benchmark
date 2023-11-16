/*
 * Copyright 2012, Haiku, Inc.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 * 		Aaron Hill <serac@hillvisions.com>
 *		Alexander von Gluck <kallisti5@unixzen.com>
 */


#include "GLifeView.h"

#include <GL/glu.h>
#include <GLView.h>
#include <math.h>
#include <stdlib.h>

#include "GLifeGrid.h"
#include "GLifeState.h"


// ------------------------------------------------------
//  GLifeView Class Constructor Definition
GLifeView::GLifeView(BRect rect, const char* name, ulong resizingMode,
	ulong options, GLifeState* pglsState)
	:
	BGLView(rect, name, resizingMode, B_FRAME_EVENTS | B_WILL_DRAW, options),
	m_pglsState(pglsState)
{
	// Setup the grid
	m_pglgGrid = new GLifeGrid(pglsState->GridWidth(), pglsState->GridHeight());
}


// ------------------------------------------------------
//  GLifeView Class Destructor Definition
GLifeView::~GLifeView(void)
{
	delete m_pglgGrid;
}


// ------------------------------------------------------
//  GLifeView Class AttachedToWindow Definition
void
GLifeView::AttachedToWindow(void)
{
	LockGL();
	BGLView::AttachedToWindow();
	
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
#if 0	
	glShadeModel(GL_SMOOTH);
#endif
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, Bounds().Width() / Bounds().Height(), 2.0, 20000.0);
	glTranslatef(0.0, 0.0, -50.0);
	glMatrixMode(GL_MODELVIEW);
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	UnlockGL();
}


// ------------------------------------------------------
//  GLifeView Class Draw Definition
void
GLifeView::Draw(BRect updateRect)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	// TODO:  Dynamic colors or user-specified coloring
	GLfloat glfGreen[] = {0.05, 0.8, 0.15, 1.0};
	GLfloat glfOrange[] = {0.65, 0.3, 0.05, 1.0};
	
	// Border control
	bool bColor;
	
	int32 iWidth = m_pglsState->GridWidth();
	int32 iHeight = m_pglsState->GridHeight();
	int32 iBorder = m_pglsState->GridBorder();
		
	glPushMatrix();
	
	glRotatef(m_glfDelta * 3, 1.0, 0.0, 0.0);
	glRotatef(m_glfDelta * 1, 0.0, 0.0, 1.0);
	glRotatef(m_glfDelta * 2, 0.0, 1.0, 0.0);
	
	for(int32 iRow = (0 - iBorder); iRow < (iHeight + iBorder); ++iRow) {
		GLfloat glfY = (GLfloat)iRow - ((GLfloat)iHeight / 2);
		
		for(int32 iColumn = (0 - iBorder); iColumn < (iWidth + iBorder); ++iColumn) {

			GLfloat glfX = (GLfloat)iColumn - ((GLfloat) iWidth / 2);
			
			bColor = (iColumn < 0) || (iColumn >= iWidth) || (iRow < 0) || (iRow >= iHeight);

			if (m_pglgGrid->Occupied(iRow, iColumn)) {
				glPushMatrix();
				
				glTranslatef(glfX, glfY, 0.0);
				glScalef(0.45, 0.45, 0.45);
			
				// GL Begin
				glBegin(GL_QUAD_STRIP);
				if (bColor)
					glColor3f( 0.65, 0.3, 0.05 );
				else
					glColor3f( 0.05, 0.8, 0.15 );

				glMaterialfv(GL_FRONT, GL_DIFFUSE, bColor ? glfOrange : glfGreen);
				glNormal3f(0.0,  1.0, 0.0);
				glVertex3f(1.0,  1.0, -1.0);
				glVertex3f(-1.0,  1.0, -1.0);
				glVertex3f(1.0,  1.0, 1.0);
				glVertex3f(-1.0,  1.0, 1.0);	
	
				glNormal3f(0.0, 0.0, 1.0);
				glVertex3f(-1.0, 1.0, 1.0);
				glVertex3f(1.0, 1.0, 1.0);
				glVertex3f(-1.0, -1.0, 1.0);
				glVertex3f(1.0, -1.0, 1.0);
			
				glNormal3f(0.0, -1.0, 0.0);
				glVertex3f(-1.0, -1.0, 1.0);
				glVertex3f(1.0, -1.0, 1.0);
				glVertex3f(-1.0, -1.0, -1.0);
				glVertex3f(1.0, -1.0, -1.0);
				glEnd();
				// GL End
			
				// GL Begin
				glBegin( GL_QUAD_STRIP);
				if (bColor)
					glColor3f(0.65, 0.3, 0.05);
				else
					glColor3f(0.05, 0.8, 0.15);

				glMaterialfv(GL_FRONT, GL_DIFFUSE, bColor ? glfOrange : glfGreen);
				glNormal3f(-1.0, 0.0, 0.0);
				glVertex3f(-1.0, 1.0, 1.0);
				glVertex3f(-1.0, -1.0, 1.0);
				glVertex3f(-1.0, 1.0, -1.0);
				glVertex3f(-1.0, -1.0, -1.0);

				glNormal3f(0.0, 0.0, -1.0);
				glVertex3f(-1.0, 1.0, -1.0);
				glVertex3f(-1.0, -1.0, -1.0);
				glVertex3f(1.0, 1.0, -1.0);
				glVertex3f(1.0, -1.0, -1.0);
	
				glNormal3f(1.0, 0.0, 0.0);
				glVertex3f(1.0, 1.0, -1.0);
				glVertex3f(1.0, -1.0, -1.0);
				glVertex3f(1.0, 1.0, 1.0);
				glVertex3f(1.0, -1.0, 1.0);
				glEnd();
				// GL End

				glPopMatrix();
			}
		}
	}
	
	glPopMatrix();
}


// ------------------------------------------------------
//  GLifeView Class Advance Definition
void
GLifeView::Advance(void)
{
	if (m_glfDelta++ > 360.0)
		m_glfDelta -= 360.0;

	int32 gridDelay = m_pglsState->GridDelay();
	if (m_iStep++ > gridDelay) {
		m_iStep = 0;
		m_pglgGrid->Generation();
	}

	LockGL();
	BRect location(0,0,0,0);
	Draw(location);
	SwapBuffers();
	UnlockGL();
}
